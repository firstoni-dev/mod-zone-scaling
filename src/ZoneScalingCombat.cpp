/*
 * Copyright (C) 2026+ mod-zone-scaling contributors, released under GNU AGPL v3
 * license: https://github.com/azerothcore/azerothcore-wotlk/blob/master/LICENSE-AGPL3
 *
 * Per-player combat normalization: the point at which this module stops
 * observing and starts changing fights.
 *
 * The model in one paragraph. A creature's real numbers stay canonical -- one
 * health pool, one set of stats, shared by everyone who fights it. What changes
 * is the damage crossing between that creature and each player, scaled so that
 * the fight *feels* like one against a creature of the player's own level. Two
 * players thirty levels apart can therefore fight the same creature at the same
 * time, each at their own difficulty, without the creature having to be two
 * different creatures.
 */

#include "ZoneScaling.h"

#include "Creature.h"
#include "CreatureData.h"
#include "Log.h"
#include "Player.h"
#include "ScriptMgr.h"
#include "SpellInfo.h"
#include "UnitScript.h"

#include <algorithm>
#include <cmath>

namespace
{
    // The player whose level this unit's damage should be measured against.
    // Pets, guardians, totems and charmed creatures answer with their owner, so
    // a hunter's pet fights at the hunter's ratio rather than at its own level
    // or at none. Returns nullptr for anything not ultimately driven by a
    // player, which is the signal to leave the damage alone.
    Player const* ResolvePlayer(Unit* unit)
    {
        return unit ? unit->GetCharmerOrOwnerPlayerOrPlayerItself() : nullptr;
    }

    // The creature side of a pairing: one that this module is actually tracking
    // and that is scalable where it stands. A creature owned by a player is
    // excluded here as well as at tagging time -- it is the player's, and it
    // has already been resolved to that player by ResolvePlayer.
    Creature* ResolveScaledCreature(Unit* unit)
    {
        if (!unit || !unit->IsCreature())
            return nullptr;

        Creature* creature = unit->ToCreature();
        if (creature->GetCharmerOrOwnerPlayerOrPlayerItself())
            return nullptr;

        ZoneScalingCreatureState const* state = ZoneScalingMgr::GetState(creature);
        return (state && state->Scalable && state->StatsValid) ? creature : nullptr;
    }
}

uint32 ZoneScalingMgr::ApplyRatio(uint32 amount, float ratio)
{
    if (!amount || ratio == 1.0f)
        return amount;

    float const scaled = float(amount) * ratio;

    // A hit that was going to land for something must still land for something.
    // Truncating to 0 reads to a player as immunity, and to the core as a hit
    // that generates no threat and fires no on-damage proc -- which turns a
    // slow fight into an unwinnable one.
    return std::max<uint32>(1, uint32(std::lround(scaled)));
}

ZoneScalingRatios ZoneScalingMgr::RatiosFrom(ZoneScalingStats const& canonical,
                                             ZoneScalingStats const& effective)
{
    ZoneScalingRatios ratios;

    // Incoming: the damage curve. Average of the melee range rather than either
    // end, because SelectLevel derives max from min by a fixed factor, so the
    // average carries the same information and is less sensitive to rounding.
    float const canonDamage = (canonical.MinDamage + canonical.MaxDamage) * 0.5f;
    float const effDamage   = (effective.MinDamage + effective.MaxDamage) * 0.5f;

    if (canonDamage > 0.0f && effDamage > 0.0f)
        ratios.Incoming = std::clamp(effDamage / canonDamage, MinRatio, MaxRatio);

    // Outgoing: the health curve, inverted. What a player observes is the
    // fraction of the bar removed per hit. The real pool is hp(canonical), the
    // perceived pool must be hp(effective), so a hit for d has to land for
    // d * hp(canonical) / hp(effective) to remove the right fraction.
    if (canonical.Health > 0 && effective.Health > 0)
        ratios.Outgoing = std::clamp(float(canonical.Health) / float(effective.Health),
                                     MinRatio, MaxRatio);

    return ratios;
}

Optional<ZoneScalingRatios> ZoneScalingMgr::DescribeRatios(CreatureTemplate const* cinfo,
                                                           uint32 zoneId, uint8 canonicalLevel,
                                                           uint8 playerLevel,
                                                           uint8& effectiveLevel) const
{
    if (!cinfo || !canonicalLevel)
        return std::nullopt;

    // Rank from the template, so `.zs ratio` predicts what the live path will
    // actually do for an elite rather than what it would do for a normal mob
    // of the same level. A diagnostic that quietly drops an input is worse than
    // no diagnostic -- it disagrees with the game and looks authoritative.
    Optional<uint8> const effective =
        GetEffectiveLevel(zoneId, playerLevel, canonicalLevel, cinfo->rank);

    if (!effective)
        return std::nullopt;

    effectiveLevel = *effective;

    if (effectiveLevel == canonicalLevel)
        return ZoneScalingRatios{};

    ZoneScalingStats canonicalStats;
    ZoneScalingStats effectiveStats;

    if (!ComputeStats(cinfo, canonicalLevel, canonicalStats)
        || !ComputeStats(cinfo, effectiveLevel, effectiveStats))
        return std::nullopt;

    return RatiosFrom(canonicalStats, effectiveStats);
}

Optional<uint8> ZoneScalingMgr::GetEffectiveLevelFor(Creature* creature, Player const* player) const
{
    if (!IsActive() || !creature || !player)
        return std::nullopt;

    ZoneScalingCreatureState const* state = GetState(creature);
    if (!state || !state->Scalable || !state->StatsValid)
        return std::nullopt;

    Optional<uint8> const effective =
        GetEffectiveLevel(state->ZoneId, player->GetLevel(), state->CanonicalLevel, state->Rank);

    // No band, or the direction rule left the creature where it was. Under
    // Direction = up this is the whole of the "creature above the player" case,
    // and it costs one comparison rather than a special path.
    if (!effective || *effective == state->CanonicalLevel)
        return std::nullopt;

    // Chrono Mode (Z-08), and the only place it is tested. This function is the
    // sole gate through which a (creature, player) pair becomes a scaled pair --
    // damage, the melee hit table and the per-observer level display all reach
    // it -- so one `return std::nullopt` here opts a player out of all three at
    // once, with no branch anywhere else that could later disagree with it.
    //
    // Tested *after* the no-op check on purpose: the counter is meant to say
    // how many fights were really left stock by a player's choice, and counting
    // pairings that would not have moved anyway would inflate it with noise
    // from every creature already at the player's level.
    if (!IsScalingActiveFor(player, state->ZoneId))
    {
        CountChronoSuppressed();
        return std::nullopt;
    }

    // Cheap enough for the hot path: EliteOffsetFor returns on the rank switch
    // for every normal creature, which is nearly all of them, before it looks
    // anything up.
    //
    // Moved below the two tests above in Z-08. It previously counted every
    // elite the arithmetic touched, including pairings that then resolved to a
    // no-op; it now counts elites in fights that were actually scaled, which is
    // the number the count was always meant to convey.
    if (EliteOffsetFor(state->ZoneId, state->CanonicalLevel, state->Rank))
        CountEliteOffset();

    return effective;
}

Optional<uint8> ZoneScalingMgr::GetViewLevelFor(Creature const* creature, Player const* player) const
{
    if (!IsActive() || !creature || !player)
        return std::nullopt;

    // An owned creature is the player's own and resolves to them, never to a
    // view -- the same exclusion ResolveScaledCreature makes for the hooks.
    if (creature->GetCharmerOrOwnerPlayerOrPlayerItself())
        return std::nullopt;

    ZoneScalingCreatureState const* state = GetState(const_cast<Creature*>(creature));
    if (!state || !state->Scalable || !state->StatsValid)
        return std::nullopt;

    Optional<uint8> const effective =
        GetEffectiveLevel(state->ZoneId, player->GetLevel(), state->CanonicalLevel, state->Rank);
    if (!effective || *effective == state->CanonicalLevel)
        return std::nullopt;

    if (!IsScalingActiveFor(player, state->ZoneId))
        return std::nullopt;

    return effective;
}

uint32 ZoneScalingMgr::GetViewArmorFor(Creature const* creature, Player const* player) const
{
    Optional<uint8> const effective = GetViewLevelFor(creature, player);
    if (!effective)
        return 0;

    ZoneScalingStats stats;
    if (!ComputeStats(creature->GetCreatureTemplate(), *effective, stats))
        return 0;

    return uint32(stats.Armor);
}

bool ZoneScalingMgr::GetViewPool(Creature const* creature, uint8 effective, ViewPool& out) const
{
    if (!creature)
        return false;

    ZoneScalingCreatureState const* state = GetState(const_cast<Creature*>(creature));
    if (!state || !state->StatsValid || !state->CanonicalStats.Health)
        return false;

    ZoneScalingStats stats;
    if (!ComputeStats(creature->GetCreatureTemplate(), effective, stats))
        return false;

    uint32 const realMax = std::max<uint32>(creature->GetMaxHealth(), 1);
    out.MaxHealth = std::max<uint32>(1, uint32(uint64(realMax) * stats.Health / state->CanonicalStats.Health));
    out.Health    = uint32(uint64(creature->GetHealth()) * out.MaxHealth / realMax);
    // Never show a creature with health left at 0: the client treats health 0 as death (CGUnit_C::OnDeath), so a
    // mob held at 1 HP by an invincibility floor (142 real, 84 shown -> 0) played its death and slid home as a corpse.
    if (!out.Health && creature->GetHealth())
        out.Health = 1;

    out.MaxMana = out.Mana = 0;
    uint32 const realMaxMana = creature->GetMaxPower(POWER_MANA);
    if (realMaxMana && state->CanonicalStats.Mana && stats.Mana)
    {
        out.MaxMana = std::max<uint32>(1, uint32(uint64(realMaxMana) * stats.Mana / state->CanonicalStats.Mana));
        out.Mana    = uint32(uint64(creature->GetPower(POWER_MANA)) * out.MaxMana / realMaxMana);
    }
    return true;
}

ZoneScalingRatios ZoneScalingMgr::GetRatios(Creature* creature, Player const* player) const
{
    ZoneScalingRatios const identity;

    if (!_scaleDamage)
        return identity;

    Optional<uint8> const effective = GetEffectiveLevelFor(creature, player);
    if (!effective)
        return identity;

    ZoneScalingCreatureState const* state = GetState(creature);

    // Only the effective-level stats are computed here; the canonical ones were
    // cached at tag time. Recomputing them per swing would double the cost of
    // the hot path for a value that cannot have changed without OnSelectLevel
    // firing and refreshing it.
    ZoneScalingStats effectiveStats;
    if (!ComputeStats(creature->GetCreatureTemplate(), *effective, effectiveStats))
        return identity;

    return RatiosFrom(state->CanonicalStats, effectiveStats);
}

namespace
{
    // The one place damage is actually changed. Both directions land here
    // because the pairing test is the same either way -- exactly one side must
    // resolve to a scaled creature and the other to a player.
    template <typename T>
    void ScaleDamage(Unit* target, Unit* attacker, T& damage, char const* what)
    {
        if (damage <= 0 || !sZoneScalingMgr->IsDamageScalingEnabled())
            return;

        Creature* creature = nullptr;
        Player const* player = nullptr;
        float ratio = 1.0f;

        if ((creature = ResolveScaledCreature(attacker)) != nullptr)
        {
            player = ResolvePlayer(target);
            ratio  = sZoneScalingMgr->GetRatios(creature, player).Incoming;
        }
        else if ((creature = ResolveScaledCreature(target)) != nullptr)
        {
            player = ResolvePlayer(attacker);
            ratio  = sZoneScalingMgr->GetRatios(creature, player).Outgoing;
        }

        // Creature against creature, player against player, or a pairing this
        // module does not scale. Leave it exactly alone.
        if (!creature || !player || ratio == 1.0f)
            return;

        T const before = damage;
        damage = T(ZoneScalingMgr::ApplyRatio(uint32(damage), ratio));
        sZoneScalingMgr->CountDamageScaled();

        if (sZoneScalingMgr->IsDebug())
            LOG_DEBUG("module.zonescaling",
                "dmg {} {} entry {} vs player level {}: {} -> {} (x{:.3f})",
                what, creature->GetGUID().ToString(), creature->GetEntry(),
                uint32(player->GetLevel()), uint32(before), uint32(damage), ratio);
    }
}

namespace
{
    // ResolveScaledCreature needs a mutable Unit* only because DataMap::Get is
    // non-const; nothing here writes to the creature. The melee-outcome hook
    // hands out const pointers, so the cast is confined to this one helper
    // rather than repeated at each call site.
    Creature* ResolveScaledCreatureConst(Unit const* unit)
    {
        return ResolveScaledCreature(const_cast<Unit*>(unit));
    }
}

class ZoneScalingUnitScript : public UnitScript
{
public:
    // Unlike AllCreatureScript this type does take an enabledHooks list, so the
    // registration costs nothing on the hooks not named here -- notably
    // OnUnitUpdate, which would otherwise be a virtual call per unit per tick.
    ZoneScalingUnitScript() : UnitScript("ZoneScalingUnitScript", true,
        {
            UNITHOOK_MODIFY_MELEE_DAMAGE,
            UNITHOOK_MODIFY_SPELL_DAMAGE_TAKEN,
            UNITHOOK_MODIFY_PERIODIC_DAMAGE_AURAS_TICK,
            UNITHOOK_ON_BEFORE_ROLL_MELEE_OUTCOME_AGAINST
        }) { }

    void ModifyMeleeDamage(Unit* target, Unit* attacker, uint32& damage) override
    {
        ScaleDamage(target, attacker, damage, "melee");
    }

    void ModifySpellDamageTaken(Unit* target, Unit* attacker, int32& damage,
                                SpellInfo const* /*spellInfo*/) override
    {
        ScaleDamage(target, attacker, damage, "spell");
    }

    void ModifyPeriodicDamageAurasTick(Unit* target, Unit* attacker, uint32& damage,
                                       SpellInfo const* spellInfo) override
    {
        // This hook is not damage-only. SpellAuraEffects.cpp calls it from
        // HandlePeriodicHealAurasTick as well, passing the *heal* amount under
        // the same parameter name -- so an unguarded implementation quietly
        // scales heals with the damage ratio. Positivity is the discriminator
        // the aura type would give us if the hook passed it.
        if (spellInfo && spellInfo->IsPositive())
            return;

        ScaleDamage(target, attacker, damage, "periodic");
    }

    // Scaling the damage number is not enough on its own, and finding that out
    // is the main thing Z-04 taught. Melee skill in this core is level * 5
    // (`Unit::GetMaxSkillValueForLevel`), taken from the creature's *canonical*
    // level. A level-8 creature presenting at 60 therefore swings with skill 40
    // against a player's defence 300: it misses most of the time, and its
    // multiplied damage never lands. Incoming damage-per-second ends up far
    // below a real level-60 creature's no matter how large the multiplier is.
    //
    // The four skill values are all mutable here, and dodge/parry/block are
    // derived from them *after* this hook via
    // `skillBonus = 4 * (attackerWeaponSkill - victimMaxSkillValueForLevel)`,
    // so correcting the skills corrects those three for free. Miss is the
    // exception: it was computed before the hook and has to be recomputed.
    void OnBeforeRollMeleeOutcomeAgainst(Unit const* attacker, Unit const* victim,
                                         WeaponAttackType attType,
                                         int32& attackerMaxSkillValueForLevel,
                                         int32& victimMaxSkillValueForLevel,
                                         int32& attackerWeaponSkill, int32& victimDefenseSkill,
                                         int32& /*crit_chance*/, int32& miss_chance,
                                         int32& /*dodge_chance*/, int32& /*parry_chance*/,
                                         int32& /*block_chance*/) override
    {
        if (!sZoneScalingMgr->IsHitChanceScalingEnabled())
            return;

        Creature* creature = nullptr;
        Player const* player = nullptr;
        bool creatureIsAttacker = false;

        if ((creature = ResolveScaledCreatureConst(attacker)) != nullptr)
        {
            player = ResolvePlayer(const_cast<Unit*>(victim));
            creatureIsAttacker = true;
        }
        else if ((creature = ResolveScaledCreatureConst(victim)) != nullptr)
        {
            player = ResolvePlayer(const_cast<Unit*>(attacker));
        }

        if (!creature || !player)
            return;

        Optional<uint8> const effective = sZoneScalingMgr->GetEffectiveLevelFor(creature, player);
        if (!effective)
            return;

        int32 const skill = int32(*effective) * 5;

        if (creatureIsAttacker)
        {
            attackerMaxSkillValueForLevel = skill;
            attackerWeaponSkill           = skill;
        }
        else
        {
            victimMaxSkillValueForLevel = skill;
            victimDefenseSkill          = skill;
        }

        // Recomputed from the corrected skills rather than adjusted, so this
        // stays in step with the core's own formula if it ever changes.
        miss_chance = int32(attacker->MeleeSpellMissChance(
            victim, attType, attackerWeaponSkill - victimMaxSkillValueForLevel, 0) * 100.0f);

        sZoneScalingMgr->CountHitTableFixed();
    }
};

void AddZoneScalingCombatScripts()
{
    new ZoneScalingUnitScript();
}
