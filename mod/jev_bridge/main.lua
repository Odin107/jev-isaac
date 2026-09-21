-- Jev bridge: opt-in, bounded movement/shooting for vanilla Repentance.
-- UDP replies must go to the observation sender. No save, console, or spawn calls.
local mod = RegisterMod("Jev Bridge", 1)
local game = Game()
local HOST, PORT = "127.0.0.1", 42421
-- Source age includes inference plus the hold: 33 simulation ticks at 30 Hz.
-- Hold the current action between replies, for at most 15 ticks after receipt.
local OBS_EVERY, MAX_AGE, MAX_HOLD = 1, 33, 15
local MAX_RECEIPT_AGE = 0.55
local MAX_MOVE_FRAMES, MAX_MOVE_DISTANCE = 6, 24
local FLOOR_LEASE_SECONDS, CLEAR_ARM_SECONDS = 2, 2
local BOSS_TRANSITION_SECONDS, BOSS_EARLY_FRAMES = 8, 2
local FLOOR_DESCENT_SECONDS = 10
local ITEM_ANIMATION_SECONDS, ITEM_ANIMATION_START_FRAMES = 3, 2
local MAX_PACKETS, MAX_ACTION_BYTES, MAX_OBS_BYTES = 32, 2048, 60000
local MAX_ENEMIES, MAX_PROJECTILES, MAX_HAZARDS = 64, 96, 512
local MAX_ENTITY_HAZARDS = 160
local MAX_PICKUPS, MAX_SWITCHES = 64, 32
local MAX_INVENTORY, MAX_COLLECTIBLE_ID = 128, 4096
local MAX_INTERACTIONS, INTERACTION_COOLDOWN = 512, 15
local json, socket, udp
local active, enabled, fault = false, false, nil
local session, roomId, visit, runCount = "", "", 0, 0
local runSession, armCount = "", 0
local command, sentFrames, lastAccepted = nil, {}, -1
local lastSentFrame, lastSentTime, retryFrame = -1, 0, 0
local lastPaused, status = false, "F8: enable Jev"
local lastStopReason = nil
local floorId, floorLeaseId, floorLeaseRun, floorLeaseUntil = "", nil, nil, 0
local clearArmUntil, expectedExit, floorTransition = 0, nil, nil
local transport, socketEpoch = nil, 0
local interactionLedger, interactionCount, lastInteractionFrame = {}, 0, -100
local inventoryCache, inventoryFrame, inventoryCount, inventoryTruncated = {}, -100, -1, false
local descentPermit = nil
local itemAnimationPermit, lastItemUse, minimumActionFrame = nil, nil, 0
local visitedAliasCache = {key = nil, aliases = {}}

local function resetTransport()
    transport = {received = 0, accepted = 0, rejected = 0, last_rejection = "none",
        last_received_frame = -1, last_accepted_frame = -1,
        socket_epoch = socketEpoch, receive_polls = 0, receive_timeouts = 0}
end
resetTransport()

local function rejectTransport(reason)
    transport.rejected = transport.rejected + 1
    transport.last_rejection = reason
end

local function acceptTransport(sourceFrame)
    transport.accepted = transport.accepted + 1
    transport.last_accepted_frame = sourceFrame
end

local moves = {
    none = {}, left = {left = true}, right = {right = true},
    up = {up = true}, down = {down = true},
    up_left = {up = true, left = true}, up_right = {up = true, right = true},
    down_left = {down = true, left = true}, down_right = {down = true, right = true},
}
local shoots = {none = true, left = true, right = true, up = true, down = true}
local interactions = {none = true, bomb = true, active = true, pocket = true}
local interactionActions = {
    [ButtonAction.ACTION_BOMB] = "bomb", [ButtonAction.ACTION_ITEM] = "active",
    [ButtonAction.ACTION_PILLCARD] = "pocket",
}
local actionDirections = {
    [ButtonAction.ACTION_LEFT] = {"move", "left"},
    [ButtonAction.ACTION_RIGHT] = {"move", "right"},
    [ButtonAction.ACTION_UP] = {"move", "up"},
    [ButtonAction.ACTION_DOWN] = {"move", "down"},
    [ButtonAction.ACTION_SHOOTLEFT] = {"shoot", "left"},
    [ButtonAction.ACTION_SHOOTRIGHT] = {"shoot", "right"},
    [ButtonAction.ACTION_SHOOTUP] = {"shoot", "up"},
    [ButtonAction.ACTION_SHOOTDOWN] = {"shoot", "down"},
}

local function disarm(reason)
    -- Pause/render callbacks can follow the original stop. Preserve its cause
    -- until the next manual arm instead of replacing it with a generic HUD hint.
    if enabled or not lastStopReason then lastStopReason = reason or "F8: enable Jev" end
    enabled, command, sentFrames, lastAccepted = false, nil, {}, -1
    floorLeaseId, floorLeaseRun, floorLeaseUntil = nil, nil, 0
    clearArmUntil, expectedExit, floorTransition = 0, nil, nil
    descentPermit = nil
    itemAnimationPermit, minimumActionFrame = nil, 0
    status = reason or "F8: enable Jev"
end

local function descriptorHash(desc)
    local hash = GetPtrHash(desc)
    if type(hash) == "number" and hash == hash and math.abs(hash) <= 9007199254740991
        and hash == math.floor(hash) then return hash end
    return nil
end

local function roomDimension(level)
    -- Vanilla Repentance has no GetDimension. Use the documented descriptor
    -- pointer comparison; grid indices alone are shared by three dimensions.
    local ok, value = pcall(function()
        if type(GetPtrHash) ~= "function" then return nil end
        local current = level:GetCurrentRoomDesc()
        if not current or not current.Data or type(current.SafeGridIndex) ~= "number"
            or current.SafeGridIndex ~= math.floor(current.SafeGridIndex)
            or math.abs(current.SafeGridIndex) > 32768 then return nil end
        local hash = descriptorHash(current)
        if hash == nil then return nil end
        for dimension = 0, 2 do
            local candidate = level:GetRoomByIdx(current.SafeGridIndex, dimension)
            if candidate and candidate.Data and descriptorHash(candidate) == hash then
                return dimension
            end
        end
    end)
    if ok then return value end
    return nil
end

local function floorState()
    local level = game:GetLevel()
    local stage, stageType, seed = level:GetStage(), level:GetStageType(), level:GetDungeonPlacementSeed()
    local current = level:GetCurrentRoomDesc()
    return {id = table.concat({stage, stageType, seed}, ":"), stage = stage,
        stage_type = stageType, seed = seed, room_index = level:GetCurrentRoomIndex(),
        room_list_index = current and current.ListIndex or nil, dimension = roomDimension(level)}
end

local function floorLeaseAlive(now, id)
    return floorLeaseId == (id or floorId) and floorLeaseRun == runSession
        and now >= floorLeaseUntil - FLOOR_LEASE_SECONDS and now < floorLeaseUntil
end

local function transitionAlive(now, id)
    if not floorTransition or now >= floorTransition.deadline then return false end
    if not floorTransition.boss then return floorLeaseAlive(now, id) end
    return floorLeaseId == (id or floorId) and floorLeaseRun == runSession
        and now >= floorTransition.startedAt
end

local function descentAlive(now)
    return descentPermit ~= nil and descentPermit.run == runSession
        and now >= descentPermit.startedAt and now < descentPermit.deadline
        and game:GetNumPlayers() == 1 and not Isaac.GetPlayer(0):IsDead()
        and (floorId == descentPermit.origin or floorId == descentPermit.destination)
end

local function descentArrival(nextFloor, now)
    return descentAlive(now) and not descentPermit.arrived
        and nextFloor.id ~= descentPermit.origin and nextFloor.stage == descentPermit.stage + 1
        and game:GetRoom():GetType() == 1 and game:GetRoom():IsClear()
end

local function floorPermissionAlive(now, id)
    return floorLeaseAlive(now, id) or transitionAlive(now, id)
        or (descentAlive(now) and (id == nil or id == floorId))
end

local function itemAnimationAlive(now)
    local permit = itemAnimationPermit
    return permit ~= nil and enabled and permit.session == session and permit.room == roomId
        and permit.floor == floorId and now >= permit.startedAt and now < permit.deadline
        and game:GetNumPlayers() == 1 and not Isaac.GetPlayer(0):IsDead()
end

local function updateItemAnimation(now)
    local permit = itemAnimationPermit
    if not permit then return end
    if not itemAnimationAlive(now) then
        if permit.paused then disarm("Item animation expired: F8 enables Jev"); return true
        else itemAnimationPermit = nil end
    elseif permit.paused and not game:IsPaused() and not permit.resumed then
        -- The first update can precede the unpause render. Restore only the
        -- short controller lease; all pre-animation inputs have been discarded.
        permit.resumed = true
        if permit.floorMode then
            floorLeaseId, floorLeaseRun, floorLeaseUntil = floorId, runSession, now + FLOOR_LEASE_SECONDS
        end
    elseif not permit.paused and (now - permit.startedAt > MAX_RECEIPT_AGE
        or game:GetFrameCount() - permit.frame > ITEM_ANIMATION_START_FRAMES) then
        itemAnimationPermit = nil
    end
end

local function closeSocket()
    if udp then pcall(function() udp:close() end) end
    udp = nil
end

local function fail(reason, detail)
    disarm(reason)
    closeSocket()
    fault = reason
    retryFrame = game:GetFrameCount() + 90
    if detail then Isaac.DebugString("[Jev] " .. tostring(detail):sub(1, 400)) end
end

local function openSocket()
    closeSocket()
    local okJson, loadedJson = pcall(require, "json")
    local okSocket, loadedSocket = pcall(require, "socket")
    if not okJson or not okSocket then
        fault = "Jev unavailable: add --luadebug; restart game"
        disarm(fault)
        return false
    end
    json, socket = loadedJson, loadedSocket
    local candidate, err = socket.udp()
    if not candidate then fail("Jev UDP unavailable; F8 retries", err); return false end
    udp = candidate
    local ok, why = udp:settimeout(0)
    if not ok then fail("Jev timeout setup failed", why); return false end
    -- Bind explicitly to loopback; connected UDP filters replies by sender IP/port.
    ok, why = udp:setsockname(HOST, 0)
    if not ok then fail("Jev loopback setup failed", why); return false end
    ok, why = udp:setpeername(HOST, PORT)
    if not ok then fail("Jev peer setup failed", why); return false end
    socketEpoch = socketEpoch + 1
    transport.socket_epoch = socketEpoch
    fault = nil
    return true
end

local function newRoom()
    itemAnimationPermit, minimumActionFrame = nil, 0
    resetTransport()
    visit = visit + 1
    local level = game:GetLevel()
    local nextFloor = floorState()
    local now = socket and socket.gettime() or 0
    local descending = enabled and descentArrival(nextFloor, now)
    local repeatedArrival = enabled and descentAlive(now) and descentPermit.arrived
        and nextFloor.id == descentPermit.destination and nextFloor.room_index == descentPermit.room
        and game:GetRoom():GetFrameCount() <= BOSS_EARLY_FRAMES
    -- A boss introduction may freeze both callbacks before the new room is
    -- reported. Accept only the commanded boss door, with at most two elapsed
    -- simulation frames and a fixed wall-clock bound from that last command.
    local bossArrival = expectedExit and expectedExit.targetType == 5
        and game:GetRoom():GetType() == 5 and expectedExit.target == nextFloor.room_index
        and game:GetFrameCount() >= expectedExit.frame
        and game:GetFrameCount() - expectedExit.frame <= BOSS_EARLY_FRAMES
        and now >= expectedExit.at and now < expectedExit.at + BOSS_TRANSITION_SECONDS
        and floorLeaseId == nextFloor.id and floorLeaseRun == runSession
    local expected = floorTransition and not floorTransition.entered and floorTransition.target
        or (expectedExit and (now - expectedExit.at <= MAX_RECEIPT_AGE or bossArrival) and expectedExit.target)
    local continuing = enabled and (floorPermissionAlive(now, nextFloor.id) or bossArrival)
        and game:GetNumPlayers() == 1 and not Isaac.GetPlayer(0):IsDead()
        and expected == nextFloor.room_index
    local boss = (floorTransition and floorTransition.boss)
        or (expectedExit and expectedExit.targetType == 5 and game:GetRoom():GetType() == 5)
    if continuing and boss and game:GetRoom():GetType() ~= 5 then continuing = false end
    local transitionStart = floorTransition and floorTransition.startedAt or (expectedExit and expectedExit.at) or now
    local transitionDeadline = floorTransition and floorTransition.deadline
        or (boss and transitionStart + BOSS_TRANSITION_SECONDS or floorLeaseUntil)
    floorId = nextFloor.id
    roomId = table.concat({level:GetStage(), level:GetStageType(),
        level:GetCurrentRoomIndex(), visit}, ":")
    lastSentFrame = -1
    -- No command or observed source frame may cross a room boundary.
    command, sentFrames, lastAccepted, expectedExit, clearArmUntil = nil, {}, -1, nil, 0
    if descending or repeatedArrival then
        descentPermit.arrived, descentPermit.destination, descentPermit.room = true, nextFloor.id, nextFloor.room_index
        floorLeaseId, floorLeaseRun, floorLeaseUntil = floorId, runSession, now + FLOOR_LEASE_SECONDS
        floorTransition = nil
        status = "Jev floor - descending; waiting for fresh controls"
    elseif continuing then
        floorTransition = {entered = true, boss = boss == true, startedAt = transitionStart,
            deadline = transitionDeadline, target = nextFloor.room_index}
        status = "Jev floor - entering room"
    else disarm("New room: F8 enables Jev") end
end

local function point(vector)
    return {x = vector.X, y = vector.Y}
end

local function entityState(entity)
    return {
        id = tostring(entity.InitSeed) .. ":" .. tostring(entity.Index),
        type = entity.Type, variant = entity.Variant, subtype = entity.SubType,
        x = entity.Position.X, y = entity.Position.Y,
        vx = entity.Velocity.X, vy = entity.Velocity.Y, radius = entity.Size,
    }
end

-- Do not depend on a JSON library's representation of empty Lua tables.
local function encodeArray(items)
    local encoded = {}
    for i, item in ipairs(items) do encoded[i] = json.encode(item) end
    return "[" .. table.concat(encoded, ",") .. "]"
end

local function configText(value, limit)
    -- Repentance may return localization keys instead of translated prose.
    -- Preserve those observed keys and keep datagrams bounded.
    return type(value) == "string" and value:sub(1, limit) or ""
end

local function observedInteger(value, lower, upper)
    if type(value) == "number" and value == value and value >= lower and value <= upper
        and value == math.floor(value) then return value end
    return nil
end

local function optionalRead(callback)
    -- Additional metadata must not disable the existing bridge on older APIs.
    -- Omitted or invalid values never grant a destructive-action permission.
    local ok, value = pcall(callback)
    if ok then return value end
    return nil
end

local function observedBombFlags(player)
    return optionalRead(function()
        local flags = player:GetBombFlags()
        -- Repentance can return BitSet128 userdata, which the game's JSON
        -- encoder silently omits. Never discard its high half or infer zero.
        if type(flags) == "userdata" then
            if observedInteger(flags.h, 0, 0) ~= 0 then return nil end
            flags = flags.l
        end
        local value = observedInteger(flags, 0, 9007199254740991)
        -- Emit an integer even when an older API returns a Lua float such as 0.0.
        return value and math.floor(value) or nil
    end)
end

local function visitedRooms(floor)
    if floor.dimension == nil then return nil end
    return optionalRead(function()
        local level, result, byPointer, seenList = game:GetLevel(), {}, {}, {}
        local rooms = level:GetRooms()
        local size = rooms and observedInteger(rooms.Size, 0, 1024)
        if not size then return nil end
        local signature = {runSession, floor.id, floor.dimension, visit, size}
        for i = 0, size - 1 do
            local desc = rooms:Get(i)
            local visited = desc and observedInteger(desc.VisitedCount, 1, 2147483647)
            -- Never export unvisited descriptors, including mapped room icons.
            if visited then
                local index = observedInteger(desc.SafeGridIndex, 0, 168)
                local list = observedInteger(desc.ListIndex, 0, 2147483647)
                local kind = desc.Data and observedInteger(desc.Data.Type, 1, 31)
                if index and list and kind and type(desc.Clear) == "boolean" then
                    local current = level:GetRoomByIdx(index, -1)
                    local pointer = descriptorHash(desc)
                    if pointer ~= nil and current and current.Data and descriptorHash(current) == pointer then
                        if #result >= 169 or seenList[list] or byPointer[pointer] then return nil end
                        local item = {room_index = index, list_index = list, type = kind,
                            clear = desc.Clear, visited_count = visited, room_indices = {}}
                        result[#result + 1], byPointer[pointer], seenList[list] = item, item, true
                        signature[#signature + 1] = table.concat({list, index, pointer}, ":")
                    end
                end
            end
        end
        -- Ask the engine for occupied quadrants. GridIndex includes the empty
        -- corner of an L room, so deriving aliases from a rectangle is unsafe.
        local cacheKey = table.concat(signature, "|")
        if visitedAliasCache.key == cacheKey then
            for pointer, item in pairs(byPointer) do
                item.room_indices = visitedAliasCache.aliases[pointer]
            end
        else
            for index = 0, 168 do
                local desc = level:GetRoomByIdx(index, -1)
                local item = desc and desc.Data and byPointer[descriptorHash(desc)]
                if item then item.room_indices[#item.room_indices + 1] = index end
            end
        end
        for _, item in ipairs(result) do
            local canonical = false
            for _, index in ipairs(item.room_indices) do
                if index == item.room_index then canonical = true end
            end
            if not canonical or #item.room_indices > 4 then return nil end
        end
        local aliases = {}
        for pointer, item in pairs(byPointer) do aliases[pointer] = item.room_indices end
        visitedAliasCache = {key = cacheKey, aliases = aliases}
        return result
    end)
end

local function gridVariant(grid)
    return observedInteger(optionalRead(function() return grid:GetVariant() end), 0, 100000) or -1
end

local function configMetadata(target, config)
    target.name = config and configText(config.Name, 80) or ""
    target.description = config and configText(config.Description, 160) or ""
end

local function pillMetadata(target, color, player)
    target.pill_known = color > 0 and game:GetItemPool():IsPillIdentified(color) or false
    -- GetPillEffect can reveal unknown pills. Never query it until identified.
    if target.pill_known then
        target.pill_effect = game:GetItemPool():GetPillEffect(color, player)
        local config = Isaac.GetItemConfig():GetPillEffect(target.pill_effect)
        target.name = config and configText(config.Name, 80) or ""
    end
end

local function playerInventory(player, frame)
    local count = player:GetCollectibleCount()
    if frame < inventoryFrame or frame - inventoryFrame >= 30 or count ~= inventoryCount then
        inventoryCache, inventoryTruncated = {}, false
        local config = Isaac.GetItemConfig()
        local maximum = math.min(MAX_COLLECTIBLE_ID, config:GetCollectibles().Size - 1)
        inventoryTruncated = config:GetCollectibles().Size - 1 > MAX_COLLECTIBLE_ID
        for id = 1, maximum do
            local owned = player:GetCollectibleNum(id, true)
            if owned > 0 then
                if #inventoryCache < MAX_INVENTORY then
                    local itemConfig = config:GetCollectible(id)
                    local item = {id = id, count = owned, type = itemConfig and itemConfig.Type or 0}
                    configMetadata(item, itemConfig)
                    inventoryCache[#inventoryCache + 1] = item
                else inventoryTruncated = true; break end
            end
        end
        inventoryFrame, inventoryCount = frame, count
    end
    return inventoryCache, inventoryTruncated
end

local function commandAlive(current, frame, now)
    return current and frame < current.expires and frame - current.frame <= MAX_AGE
        and now - current.startedAt <= MAX_RECEIPT_AGE
end

-- Short movements stop locally while the bounded shooting command continues.
-- Once stopped, a command cannot resume movement if the player drifts back.
local function appliedMove(current, frame, now, player)
    if current.move == "none" or not current.move_frames then return current.move end
    if not current.moveStopReason then
        if frame >= current.moveExpires or now >= current.moveStartedAt + current.move_frames / 30 then
            current.moveStopReason = "time_limit"
        else
            local dx = player.Position.X - current.moveOriginX
            local dy = player.Position.Y - current.moveOriginY
            if dx * dx + dy * dy >= current.move_distance * current.move_distance then
                current.moveStopReason = "distance_limit"
            end
        end
    end
    return current.moveStopReason and "none" or current.move
end

-- Engine EffectVariant values, distinct from PLAYER_CREEP variants. These
-- include damaging and movement-affecting surfaces; raw damage/scale remain
-- observations, not a claim that every colored patch damages this character.
local hostileCreep = {[22] = "red", [23] = "green", [24] = "yellow",
    [25] = "white", [26] = "black", [56] = "brown", [94] = "slippery brown",
    [155] = "growing slippery brown", [169] = "static", [195] = "liquid poop"}

local function observation()
    local player, room = Isaac.GetPlayer(0), game:GetRoom()
    local enemies, projectiles, hazards, doors, pickups, switches = {}, {}, {}, {}, {}, {}
    local truncated = {enemies = false, projectiles = false, hazards = false, pickups = false, switches = false}
    local function append(items, limit, kind, item)
        if #items < limit then items[#items + 1] = item else truncated[kind] = true end
    end
    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:Exists() and not entity:IsDead() then
            -- Movable TNT is an NPC object, but it is an obstacle rather than a
            -- combat target. Keep it distinct from a dropped bomb with a fuse.
            if entity.Type == EntityType.ENTITY_MOVABLE_TNT then
                local item = entityState(entity)
                item.kind = "tnt"
                append(hazards, MAX_ENTITY_HAZARDS, "hazards", item)
            elseif entity:ToNPC() and entity:IsActiveEnemy(false)
                and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY)
                and not entity:HasEntityFlags(EntityFlag.FLAG_CHARM) then
                local item = entityState(entity)
                item.hp, item.max_hp = entity.HitPoints, entity.MaxHitPoints
                item.vulnerable = entity:IsVulnerableEnemy()
                -- Call the base Entity method: EntityNPC.CanShutDoors is a
                -- boolean field with the same name after casting to ToNPC().
                item.keeps_doors_closed = entity:CanShutDoors()
                append(enemies, MAX_ENEMIES, "enemies", item)
            elseif entity.Type == EntityType.ENTITY_PROJECTILE then
                append(projectiles, MAX_PROJECTILES, "projectiles", entityState(entity))
            elseif entity.Type == EntityType.ENTITY_PICKUP and entity.Visible == true then
                local pickup = entity:ToPickup()
                if pickup then
                    local item = entityState(pickup)
                    item.price = pickup.Price
                    item.shop_item = pickup:IsShopItem()
                    item.options_index = pickup.OptionsPickupIndex
                    item.wait = pickup.Wait
                    item.collectible_kind = 0
                    if pickup.Variant == PickupVariant.PICKUP_COLLECTIBLE and pickup.SubType > 0 then
                        local config = Isaac.GetItemConfig():GetCollectible(pickup.SubType)
                        if config then
                            item.collectible_kind = config.Type
                            item.quality = config.Quality
                            configMetadata(item, config)
                        end
                    elseif pickup.Variant == PickupVariant.PICKUP_TAROTCARD and pickup.SubType > 0 then
                        configMetadata(item, Isaac.GetItemConfig():GetCard(pickup.SubType))
                    elseif pickup.Variant == PickupVariant.PICKUP_PILL then
                        pillMetadata(item, pickup.SubType, player)
                    end
                    append(pickups, MAX_PICKUPS, "pickups", item)
                end
            elseif entity.Type == EntityType.ENTITY_EFFECT and hostileCreep[entity.Variant] then
                local friendly = optionalRead(function()
                    local source = entity.SpawnerEntity
                    return source and (source:ToPlayer() ~= nil
                        or source:HasEntityFlags(EntityFlag.FLAG_FRIENDLY)
                        or source:HasEntityFlags(EntityFlag.FLAG_CHARM)) or false
                end)
                if friendly ~= true then
                    local effect = entity:ToEffect()
                    local item = entityState(entity)
                    item.kind, item.surface = "creep", hostileCreep[entity.Variant]
                    item.radius_basis = "Entity.Size; footprint estimate"
                    item.ground_effect = true
                    if effect then
                        item.timeout = observedInteger(effect.Timeout, -1, 100000)
                        item.age = observedInteger(effect.FrameCount, 0, 10000000)
                        local scale, damage = effect.Scale, effect.CollisionDamage
                        if type(scale) == "number" and scale == scale and scale >= 0 and scale <= 1000 then item.scale = scale end
                        if type(damage) == "number" and damage == damage and damage >= 0 and damage <= 100000 then item.collision_damage = damage end
                    end
                    append(hazards, MAX_ENTITY_HAZARDS, "hazards", item)
                end
            elseif entity.Type == EntityType.ENTITY_BOMBDROP
                or entity.Type == EntityType.ENTITY_LASER
                or entity.Type == EntityType.ENTITY_FIREPLACE then
                local item = entityState(entity)
                item.kind = entity.Type == EntityType.ENTITY_LASER and "laser"
                    or (entity.Type == EntityType.ENTITY_BOMBDROP and "bomb" or "fire")
                if item.kind == "fire" then
                    local hp = entity.HitPoints
                    local maxHp = entity.MaxHitPoints
                    item.hp = type(hp) == "number" and hp == hp and math.abs(hp) < 1000000 and hp or nil
                    item.max_hp = type(maxHp) == "number" and maxHp == maxHp and math.abs(maxHp) < 1000000 and maxHp or nil
                    -- Only ordinary and red fireplaces can be shot out with
                    -- ordinary tears. Blue/purple/white fires stay hazards.
                    item.tear_destructible = (entity.Variant == 0 or entity.Variant == 1)
                        and item.hp ~= nil and item.hp > 0
                end
                append(hazards, MAX_ENTITY_HAZARDS, "hazards", item)
            end
        end
    end
    for index = 0, room:GetGridSize() - 1 do
        local grid = room:GetGridEntity(index)
        local collision = room:GetGridCollision(index)
        local gridType = grid and grid:GetType() or 0
        if gridType == 20 then
            -- Pressure plates have COLLISION_NONE and must be observed outside
            -- the hazard filter. Raw variant/state distinguish required room
            -- plates from reward, Greed, rail and other special switches.
            local item = optionalRead(function()
                local pos = room:GetGridPosition(index)
                if not observedInteger(index, 0, 4095) or not observedInteger(collision, 0, 5)
                    or type(pos.X) ~= "number" or pos.X ~= pos.X or math.abs(pos.X) > 1000000
                    or type(pos.Y) ~= "number" or pos.Y ~= pos.Y or math.abs(pos.Y) > 1000000 then return nil end
                return {index = index, type = gridType, collision = collision, x = pos.X, y = pos.Y,
                    variant = observedInteger(optionalRead(function() return grid:GetVariant() end), 0, 100000),
                    state = observedInteger(optionalRead(function() return grid.State end), 0, 100000)}
            end)
            if item then append(switches, MAX_SWITCHES, "switches", item)
            else truncated.switches = true end
        end
        if collision ~= GridCollisionClass.COLLISION_NONE
            or gridType == GridEntityType.GRID_SPIKES or gridType == GridEntityType.GRID_SPIKES_ONOFF
            or gridType == GridEntityType.GRID_TRAPDOOR or gridType == GridEntityType.GRID_STAIRS
            or gridType == GridEntityType.GRID_TELEPORTER or gridType == 14 then
            local pos = room:GetGridPosition(index)
            local variant = gridVariant(grid)
            local state = grid and grid.State or 0
            local item = {
                kind = "grid", index = index, type = gridType, collision = collision,
                state = state, variant = variant, x = pos.X, y = pos.Y, radius = 20,
            }
            if gridType == 14 then
                -- Poop state measures damage: 1000 is completely broken.
                -- Keep its observed remainder after collision disappears so
                -- callers can distinguish completion from missing metadata.
                item.tear_destructible = variant == 0 and collision == 3
                    and observedInteger(grid and grid.State, 0, 999) ~= nil
            end
            append(hazards, MAX_HAZARDS, "hazards", item)
        end
    end
    -- Observe only doors physically present in this room, never the hidden map.
    for slot = 0, 7 do
        local door = room:GetDoor(slot)
        if door then
            local currentType = observedInteger(optionalRead(function() return door.CurrentRoomType end), 0, 31)
            local busted = optionalRead(function() return door:IsBusted() end)
            local item = {slot = slot, x = door.Position.X, y = door.Position.Y,
                open = door:IsOpen(), locked = door:IsLocked(),
                target_index = door.TargetRoomIndex, target_type = door.TargetRoomType,
                current_type = currentType or -1, variant = gridVariant(door)}
            if type(busted) == "boolean" then item.busted = busted end
            -- Room association is not proof of visible spikes or actual cost:
            -- Flat File, secret passages and other effects may change those.
            if currentType == 10 or door.TargetRoomType == 10 then item.curse_room_door = true
            elseif currentType ~= nil and observedInteger(door.TargetRoomType, 0, 31) ~= nil then
                item.curse_room_door = false
            end
            doors[#doors + 1] = item
        end
    end
    local playerState = entityState(player)
    playerState.hearts = player:GetHearts()
    playerState.soul_hearts = player:GetSoulHearts()
    playerState.max_hearts = player:GetMaxHearts()
    playerState.dead = player:IsDead()
    playerState.move_speed = player.MoveSpeed
    playerState.coins = player:GetNumCoins()
    playerState.bombs = player:GetNumBombs()
    playerState.giga_bombs = player:GetNumGigaBombs()
    playerState.bomb_flags = observedBombFlags(player)
    playerState.unsafe_bomb_trinket = player:HasTrinket(73) or player:HasTrinket(133)
    playerState.keys = player:GetNumKeys()
    playerState.can_pick_red_hearts = player:CanPickRedHearts()
    playerState.can_pick_soul_hearts = player:CanPickSoulHearts()
    playerState.can_pick_black_hearts = player:CanPickBlackHearts()
    playerState.can_pickup_items = player:CanPickupItem()
    playerState.active_item = player:GetActiveItem(0)
    playerState.active_charge = player:GetActiveCharge(0)
    local activeConfig = playerState.active_item > 0
        and Isaac.GetItemConfig():GetCollectible(playerState.active_item) or nil
    playerState.active_max_charge = activeConfig and activeConfig.MaxCharges or 0
    playerState.active_charge_type = activeConfig and activeConfig.ChargeType or 0
    playerState.active_name = activeConfig and configText(activeConfig.Name, 80) or ""
    playerState.active_description = activeConfig and configText(activeConfig.Description, 160) or ""
    playerState.pocket_card, playerState.pocket_pill = player:GetCard(0), player:GetPill(0)
    playerState.trinket = player:GetTrinket(0)
    playerState.trinket_1 = player:GetTrinket(1)
    local pocket = {}
    if playerState.pocket_card > 0 then
        configMetadata(pocket, Isaac.GetItemConfig():GetCard(playerState.pocket_card))
    else pillMetadata(pocket, playerState.pocket_pill, player) end
    playerState.pocket_name, playerState.pocket_description = pocket.name or "", pocket.description or ""
    playerState.pill_known, playerState.pill_effect = pocket.pill_known or false, pocket.pill_effect
    playerState.shot_speed, playerState.tear_range = player.ShotSpeed, player.TearRange
    playerState.damage, playerState.max_fire_delay = player.Damage, player.MaxFireDelay
    playerState.can_fly = player.CanFly == true
    playerState.player_type = observedInteger(optionalRead(function() return player:GetPlayerType() end), 0, 10000) or -1
    playerState.has_flat_file = optionalRead(function() return player:HasTrinket(151) end) == true
    local weaponTypes = {}
    for weapon = 1, 15 do
        if player:HasWeaponType(weapon) then weaponTypes[#weaponTypes + 1] = weapon end
    end
    playerState.weapon_type = weaponTypes[1] or 0
    local inventory
    inventory, playerState.inventory_truncated = playerInventory(player, game:GetFrameCount())
    local encodedPlayer = json.encode(playerState)
    encodedPlayer = encodedPlayer:sub(1, -2) .. ',"inventory":' .. encodeArray(inventory)
        .. ',"weapon_types":' .. encodeArray(weaponTypes) .. "}"
    local data = {
        protocol = 1, type = "observation", session = session, room_id = roomId, run_id = runSession,
        capabilities = {movement_pulses = 1, local_goal_control = 1, floor_control = 1,
            transport_diagnostics = 1, pickup_collection = 1, interaction_control = 1, floor_descent = 1,
            room_switches = 1, ground_creep = 1, item_animations = 1, observation_recovery = 1},
        frame = game:GetFrameCount(), enabled = enabled, paused = game:IsPaused(),
        status = status, last_stop_reason = lastStopReason, transport = transport,
        last_item_use = lastItemUse,
        item_animation = itemAnimationPermit ~= nil and itemAnimationPermit.paused == true,
        floor = floorState(), floor_mode = floorPermissionAlive(socket.gettime()),
        floor_transition = floorTransition ~= nil or descentPermit ~= nil,
        floor_advance_permitted = descentAlive(socket.gettime()),
        room = {top_left = point(room:GetTopLeftPos()), bottom_right = point(room:GetBottomRightPos()),
            clear = room:IsClear(), type = room:GetType(), shape = room:GetRoomShape(), frame = room:GetFrameCount()},
        truncated = truncated.enemies or truncated.projectiles or truncated.hazards or truncated.pickups or truncated.switches,
        truncated_arrays = truncated,
    }
    local hasTriggerPlates = optionalRead(function() return room:HasTriggerPressurePlates() end)
    if type(hasTriggerPlates) == "boolean" then data.room.has_trigger_pressure_plates = hasTriggerPlates end
    local visited = visitedRooms(data.floor)
    local encodedVisited = ""
    if visited then
        local rows = {}
        for _, item in ipairs(visited) do
            local indices = item.room_indices
            item.room_indices = nil
            local row = json.encode(item)
            rows[#rows + 1] = row:sub(1, -2) .. ',"room_indices":' .. encodeArray(indices) .. "}"
        end
        encodedVisited = ',"visited_rooms":[' .. table.concat(rows, ",") .. "]"
        data.capabilities.visited_rooms = 1
    end
    local frame, now = game:GetFrameCount(), socket.gettime()
    if enabled and not game:IsPaused() and not player:IsDead() and commandAlive(command, frame, now) then
        data.control = {requested_move = command.move,
            applied_move = appliedMove(command, frame, now, player), shoot = command.shoot,
            source_frame = command.frame, move_stop_reason = command.moveStopReason or "none",
            interaction = command.interaction, interaction_id = command.interaction_id,
            interaction_status = command.interactionStatus}
    end
    local base = json.encode(data)
    local arrays = ',"player":' .. encodedPlayer .. ',"enemies":' .. encodeArray(enemies)
        .. ',"projectiles":' .. encodeArray(projectiles) .. ',"hazards":' .. encodeArray(hazards)
        .. ',"doors":' .. encodeArray(doors) .. ',"pickups":' .. encodeArray(pickups)
    local encodedSwitches = ',"switches":' .. encodeArray(switches)
    local payload = base:sub(1, -2) .. arrays .. encodedSwitches .. encodedVisited .. "}"
    if visited and #payload > MAX_OBS_BYTES then
        -- Room history is optional. Preserve the complete fresh room packet
        -- when history alone would exceed UDP's observation budget.
        data.capabilities.visited_rooms = nil
        base = json.encode(data)
        payload = base:sub(1, -2) .. arrays .. encodedSwitches .. "}"
    end
    if #payload > MAX_OBS_BYTES then
        -- The switch extension is optional for older consumers. If its byte
        -- cost alone overflows the packet, omit it completely; never publish
        -- an empty or partial array as evidence that a switch is absent.
        data.capabilities.room_switches = nil
        data.room.has_trigger_pressure_plates = nil
        base = json.encode(data)
        payload = base:sub(1, -2) .. arrays .. "}"
    end
    return payload
end

local function sendObservation()
    if not active or not udp or game:GetNumPlayers() < 1 then return end
    local payload = observation()
    if #payload > MAX_OBS_BYTES then fail("Jev observation too large; manual control"); return end
    local sent, err = udp:send(payload)
    if not sent then fail("Jev disconnected; manual control", err); return end
    local frame = game:GetFrameCount()
    lastSentFrame, lastSentTime = frame, socket.gettime()
    if enabled and not game:IsPaused() then sentFrames[frame] = true end
    for sentFrame in pairs(sentFrames) do
        if frame - sentFrame > MAX_AGE then sentFrames[sentFrame] = nil end
    end
end

local function integer(value)
    return type(value) == "number" and value == value and value ~= math.huge
        and value ~= -math.huge and value == math.floor(value)
end

local function parseAction(payload, frame)
    if #payload > MAX_ACTION_BYTES or not payload:match("^%s*{") or not payload:match("}%s*$") then return nil, "shape" end
    -- The command schema is flat. Reject nested structures before decoding.
    -- Strings emitted by this protocol never contain literal braces or brackets.
    if payload:sub(2, -2):find("[%[%]{}]") then return nil, "nested" end
    local ok, data, nextPosition, decodeError = pcall(json.decode, payload)
    if not ok or decodeError or type(data) ~= "table" then return nil, "json" end
    if type(nextPosition) == "number" and payload:sub(nextPosition):match("%S") then return nil, "json" end
    if data.protocol ~= 1 or data.type ~= "action" then return nil, "protocol" end
    if data.session ~= session or data.room_id ~= roomId then return nil, "identity" end
    if (data.floor_mode ~= nil or payload:find('"floor_mode"%s*:')) and type(data.floor_mode) ~= "boolean" then return nil, "optional_floor" end
    if data.interaction ~= nil or payload:find('"interaction"%s*:') then
        if type(data.interaction) ~= "string" or not interactions[data.interaction] then return nil, "interaction" end
    else data.interaction = "none" end
    if data.interaction_id ~= nil or payload:find('"interaction_id"%s*:') then
        if type(data.interaction_id) ~= "string" or #data.interaction_id < 1
            or #data.interaction_id > 128 or data.interaction_id:find("[%c]") then return nil, "interaction" end
    end
    if data.interaction ~= "none" and not data.interaction_id then return nil, "interaction" end
    if data.transition ~= nil or payload:find('"transition"%s*:') then
        if data.transition ~= "floor" or data.floor_mode ~= true then return nil, "transition" end
    end
    local revoke = data.floor_mode == false and data.move == "none" and data.shoot == "none"
        and data.interaction == "none" and data.hold_frames == 1
    if data.stop_reason ~= nil or payload:find('"stop_reason"%s*:') then
        if (data.stop_reason ~= "navigation" and data.stop_reason ~= "observation") or not revoke
            or data.move_frames ~= nil or data.move_distance ~= nil
            or payload:find('"move_frames"%s*:') or payload:find('"move_distance"%s*:') then
            return nil, "stop_reason"
        end
    end
    if not integer(data.frame) then return nil, "frame_unknown" end
    if data.frame > frame then return nil, "frame_future" end
    if frame - data.frame > MAX_AGE or data.frame < lastAccepted
        or data.frame < minimumActionFrame then return nil, "frame_old" end
    if not sentFrames[data.frame] then return nil, "frame_unknown" end
    if data.frame == lastAccepted and not revoke then return nil, "frame_duplicate" end
    if not integer(data.hold_frames) or data.hold_frames < 1 or data.hold_frames > MAX_HOLD then return nil, "duration" end
    -- Both optional fields are required together. Legacy actions keep their
    -- original hold behavior; bounded movement is an explicit controller opt-in.
    if data.move_frames ~= nil or data.move_distance ~= nil
        or payload:find('"move_frames"%s*:') or payload:find('"move_distance"%s*:') then
        if not integer(data.move_frames) or data.move_frames < 1 or data.move_frames > MAX_MOVE_FRAMES
            or data.move_frames > data.hold_frames then return nil, "movement" end
        if type(data.move_distance) ~= "number" or data.move_distance ~= data.move_distance
            or data.move_distance <= 0 or data.move_distance > MAX_MOVE_DISTANCE then return nil, "movement" end
    end
    if type(data.move) ~= "string" or not moves[data.move]
        or type(data.shoot) ~= "string" or not shoots[data.shoot] then return nil, "action" end
    data.revoke = revoke
    return data
end

local function updateExpectedExit(current, now, player)
    expectedExit = nil
    local room = game:GetRoom()
    if current.floor_mode ~= true or not room:IsClear() then return end
    local outward = {"left", "up", "right", "down"}
    for slot = 0, 7 do
        local door = room:GetDoor(slot)
        if door and door:IsOpen() and not door:IsLocked()
            and moves[current.move][outward[slot % 4 + 1]] then
            local dx, dy = player.Position.X - door.Position.X, player.Position.Y - door.Position.Y
            if dx * dx + dy * dy <= 64 * 64 then
                expectedExit = {target = door.TargetRoomIndex, targetType = door.TargetRoomType,
                    frame = game:GetFrameCount(), at = now}
                return
            end
        end
    end
end

local function updateDescentPermit(current, now, player)
    -- Only a deliberate approach to an actually observed, cleared boss exit
    -- permits normal next-stage continuity. Never discover a hidden exit/map.
    if descentPermit and descentPermit.arrived then
        if not lastPaused and game:GetRoom():GetFrameCount() > BOSS_EARLY_FRAMES then descentPermit = nil end
        return
    end
    if current.transition ~= "floor" or current.floor_mode ~= true then descentPermit = nil; return end
    local room = game:GetRoom()
    if room:GetType() ~= 5 or not room:IsClear() then descentPermit = nil; return end
    for index = 0, room:GetGridSize() - 1 do
        local grid = room:GetGridEntity(index)
        if grid and (grid:GetType() == GridEntityType.GRID_TRAPDOOR
            or grid:GetType() == GridEntityType.GRID_STAIRS) then
            local pos = room:GetGridPosition(index)
            local dx, dy = player.Position.X - pos.X, player.Position.Y - pos.Y
            if dx * dx + dy * dy <= 64 * 64 then
                if not descentPermit then
                    descentPermit = {run = runSession, origin = floorId, stage = floorState().stage,
                        startedAt = now, deadline = now + FLOOR_DESCENT_SECONDS, arrived = false}
                end
                return
            end
        end
    end
    descentPermit = nil
end

local function readActions(frame)
    if not udp then return end
    local newest
    for _ = 1, MAX_PACKETS do
        -- +1 lets us reject a packet truncated at our action-size limit.
        transport.receive_polls = transport.receive_polls + 1
        local payload, err = udp:receive(MAX_ACTION_BYTES + 1)
        if not payload then
            if err == "timeout" then transport.receive_timeouts = transport.receive_timeouts + 1
            else fail("Jev disconnected; manual control", err) end
            break
        end
        -- Count actual received datagrams, even when controls are inactive.
        -- Receipt uses the current game frame; acceptance records the source frame.
        transport.received = transport.received + 1
        transport.last_received_frame = frame
        if enabled and not game:IsPaused() then
            local candidate, rejection = parseAction(payload, frame)
            if not candidate then rejectTransport(rejection) end
            if candidate and candidate.revoke then
                acceptTransport(candidate.frame)
                disarm(candidate.stop_reason == "observation" and "Incomplete room data: F8 retries"
                    or candidate.stop_reason == "navigation" and "Route blocked: F8 retries"
                    or "Controller released control"); return
            end
            -- Valid commands superseded within this receive batch are neither
            -- rejected nor applied, so the three counters need not sum exactly.
            if candidate and (not newest or candidate.frame > newest.frame) then newest = candidate end
        else
            rejectTransport("inactive")
        end
    end
    if newest and enabled and udp then
        local previous = command
        local now, player = socket.gettime(), Isaac.GetPlayer(0)
        if descentPermit and not descentAlive(now) then
            disarm("Floor descent expired: F8 enables Jev"); return
        end
        if floorTransition and not transitionAlive(now) then
            disarm("Transition expired: F8 enables Jev"); return
        end
        if newest.floor_mode == true then
            floorLeaseId, floorLeaseRun, floorLeaseUntil = floorId, runSession, now + FLOOR_LEASE_SECONDS
        else floorLeaseId, floorLeaseRun, floorLeaseUntil = nil, nil, 0 end
        clearArmUntil = 0
        -- An update may accept the first new-room packet before the render
        -- callback sees IsPaused turn false. Keep that transition evidence
        -- until the pause edge has been processed, or it would disarm again.
        if not lastPaused and not (floorTransition and floorTransition.boss
            and game:GetRoom():GetFrameCount() <= BOSS_EARLY_FRAMES) then floorTransition = nil end
        updateExpectedExit(newest, now, player)
        updateDescentPermit(newest, now, player)
        local previousMove = previous and commandAlive(previous, frame, now)
            and appliedMove(previous, frame, now, player) or "none"
        command = newest
        -- Receipt is after this simulation tick. Include the next hold_frames ticks.
        command.expires = math.min(frame + newest.hold_frames + 1, newest.frame + MAX_AGE + 1)
        command.triggerFrame = frame + 1
        command.interactionStatus = "none"
        if command.interaction ~= "none" then
            if interactionLedger[command.interaction_id] then command.interactionStatus = "duplicate"
            elseif interactionCount >= MAX_INTERACTIONS then command.interactionStatus = "limit"
            elseif frame - lastInteractionFrame < INTERACTION_COOLDOWN then command.interactionStatus = "cooldown"
            else
                -- Reserve once when accepting, even if a later pause/superseding
                -- command prevents the pulse. Packet replay can never spend twice.
                interactionLedger[command.interaction_id] = true
                interactionCount, lastInteractionFrame = interactionCount + 1, frame
                command.interactionPulse, command.interactionStatus = true, "accepted"
            end
        end
        command.startedAt = now
        command.moveStopReason = nil
        if command.move_frames then
            command.moveExpires = frame + command.move_frames + 1
            command.moveStartedAt = now
            command.moveOriginX, command.moveOriginY = player.Position.X, player.Position.Y
        end
        command.triggered = {}
        for action, direction in pairs(actionDirections) do
            local down = direction[1] == "move" and moves[newest.move][direction[2]] == true
                or direction[1] == "shoot" and newest.shoot == direction[2]
            local wasDown = false
            if previous and commandAlive(previous, frame, now) then
                wasDown = direction[1] == "move" and moves[previousMove][direction[2]] == true
                    or direction[1] == "shoot" and previous.shoot == direction[2]
            end
            command.triggered[action] = down and not wasDown
        end
        lastAccepted, status = newest.frame, "Jev ON - F8 stops"
        acceptTransport(newest.frame)
    end
end

local function guarded(callback)
    return function(...)
        local ok, result = pcall(callback, ...)
        if not ok then fail("Jev error; manual control (see log)", result); return nil end
        return result
    end
end

local function confirmItemUse(kind, item, usedBy, slot)
    if not active or not enabled or not udp or not command or game:GetNumPlayers() ~= 1 then return end
    local player, frame, now = Isaac.GetPlayer(0), game:GetFrameCount(), socket.gettime()
    local pulse = command.itemInput
    if not pulse or command.itemUseConfirmed or pulse.kind ~= kind
        or (kind ~= "pill" and pulse.item ~= item) or (kind == "active" and slot ~= 0)
        or not usedBy or usedBy.Index ~= player.Index or usedBy.InitSeed ~= player.InitSeed
        or player:IsDead() or frame ~= command.triggerFrame or not commandAlive(command, frame, now) then return end
    command.itemUseConfirmed = true
    lastItemUse = {kind = kind, item = pulse.item, frame = frame, room_id = roomId,
        interaction_id = command.interaction_id}
    itemAnimationPermit = {session = session, room = roomId, floor = floorId,
        startedAt = now, deadline = now + ITEM_ANIMATION_SECONDS, frame = frame,
        floorMode = command.floor_mode == true}
    -- Observing the callback must not change the item's animation, cost or effect.
end

mod:AddCallback(ModCallbacks.MC_USE_ITEM, guarded(function(_, item, _, player, _, slot)
    confirmItemUse("active", item, player, slot)
end))
mod:AddCallback(ModCallbacks.MC_USE_CARD, guarded(function(_, card, player)
    confirmItemUse("card", card, player)
end))
mod:AddCallback(ModCallbacks.MC_USE_PILL, guarded(function(_, _, player)
    confirmItemUse("pill", nil, player)
end))

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, guarded(function()
    lastItemUse = nil
    disarm("F8: enable Jev")
    active, runCount, visit = true, runCount + 1, 0
    interactionLedger, interactionCount, lastInteractionFrame = {}, 0, -100
    inventoryCache, inventoryFrame, inventoryCount, inventoryTruncated = {}, -100, -1, false
    openSocket()
    local stamp = socket and socket.gettime() or 0
    runSession = string.format("%.6f", stamp) .. ":" .. tostring(runCount)
        .. ":" .. tostring(game:GetSeeds():GetStartSeed()) .. ":" .. tostring({})
    armCount, session = 0, runSession .. ":arm0"
    lastPaused = game:IsPaused()
    newRoom()
    sendObservation()
end))

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, guarded(function()
    if active then newRoom(); sendObservation() end
end))

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, guarded(function()
    if not active then return end
    local frame = game:GetFrameCount()
    if not udp and socket and frame >= retryFrame then openSocket() end
    if game:GetNumPlayers() ~= 1 then
        disarm("Jev needs one player")
        if game:GetNumPlayers() < 1 then return end
    end
    local player = Isaac.GetPlayer(0)
    local now = socket and socket.gettime() or 0
    local currentFloor = floorState()
    if currentFloor.id ~= floorId then
        if enabled and descentArrival(currentFloor, now) then
            -- A reordered update may arrive before MC_POST_NEW_ROOM. Release
            -- inputs and wait for that callback to authenticate the new room.
            command, sentFrames, lastAccepted = nil, {}, -1
            return
        else disarm("Floor changed: Jev stopped") end
    end
    if player:IsDead() then disarm("Player died: manual control") end
    updateItemAnimation(now)
    if descentPermit and not descentAlive(now) then disarm("Floor descent expired: F8 enables Jev") end
    if floorTransition and not transitionAlive(now) then
        disarm("Transition expired: F8 enables Jev")
    end
    if enabled and game:GetRoom():IsClear() and not floorPermissionAlive(now)
        and not (itemAnimationAlive(now) and itemAnimationPermit.paused)
        and now >= clearArmUntil then
        local reason = clearArmUntil > 0 and "No valid controller reply: F8 retries"
            or (floorLeaseUntil > 0 and "Controller timed out: F8 retries" or "Room cleared: Jev stopped")
        disarm(reason)
    end
    if command and not commandAlive(command, frame, socket.gettime()) then
        command = nil
        status = "Jev waiting - manual control"
    end
    readActions(frame)
    if frame % OBS_EVERY == 0 and frame ~= lastSentFrame then sendObservation() end
end))

mod:AddCallback(ModCallbacks.MC_POST_RENDER, guarded(function()
    if not active then return end
    local paused = game:IsPaused()
    local now = socket and socket.gettime() or 0
    -- ACTION_PAUSE is not reliably polled by the vanilla engine. Keyboard pause
    -- keys explicitly revoke even while an anticipated door transition is pending.
    local pauseKey = Input.IsButtonTriggered(Keyboard.KEY_ESCAPE, 0)
        or Input.IsButtonTriggered(Keyboard.KEY_P, 0)
        or (game:GetNumPlayers() == 1
            and Input.IsActionTriggered(ButtonAction.ACTION_PAUSE, Isaac.GetPlayer(0).ControllerIndex))
    if pauseKey and enabled then disarm("Paused: Jev stopped"); sendObservation() end
    if updateItemAnimation(now) then sendObservation() end
    if descentPermit and not descentAlive(now) then
        disarm("Floor descent expired: F8 enables Jev"); sendObservation()
    end
    if floorTransition and not transitionAlive(now) then
        disarm("Transition expired: F8 enables Jev"); sendObservation()
    end
    if paused ~= lastPaused then
        if itemAnimationAlive(now) and paused and not itemAnimationPermit.paused then
            itemAnimationPermit.paused = true
            command, sentFrames, lastAccepted = nil, {}, -1
            minimumActionFrame = game:GetFrameCount() + 1
            expectedExit, floorTransition = nil, nil
            status = "Jev - item animation"
            Isaac.DebugString("[Jev] Confirmed item animation; controls suspended")
        elseif itemAnimationAlive(now) and not paused and itemAnimationPermit.resumed then
            itemAnimationPermit = nil
            status = "Jev - waiting for fresh controls after item"
            Isaac.DebugString("[Jev] Item animation finished; waiting for fresh controls")
        elseif enabled and descentAlive(now) then
            command, sentFrames, lastAccepted = nil, {}, -1
            if paused then status = "Jev floor - descending"
            elseif descentPermit.arrived then status = "Jev floor - waiting for fresh floor controls"
            else disarm("Floor descent canceled: F8 enables Jev") end
        elseif paused and enabled and floorPermissionAlive(now)
            and (floorTransition or (expectedExit and now - expectedExit.at <= MAX_RECEIPT_AGE)) then
            local boss = expectedExit and expectedExit.targetType == 5
            floorTransition = floorTransition or {entered = false, boss = boss == true,
                startedAt = expectedExit.at,
                deadline = boss and expectedExit.at + BOSS_TRANSITION_SECONDS or floorLeaseUntil,
                target = expectedExit.target}
            command, sentFrames, lastAccepted = nil, {}, -1
            status = "Jev floor - room transition"
        elseif not paused and enabled and floorTransition and floorTransition.entered and floorPermissionAlive(now) then
            -- Boss introductions can produce more than one pause edge during
            -- the first room frames. Keep the fixed permit until a fresh floor
            -- command arrives after those frames; no previous command survives.
            if not floorTransition.boss then floorTransition = nil end
            status = "Jev floor - waiting for fresh room controls"
        else disarm(paused and "Paused: Jev stopped" or "F8: enable Jev") end
        lastPaused = paused
        sendObservation()
    end
    if Input.IsButtonTriggered(Keyboard.KEY_F8, 0) then
        if enabled then disarm("Jev OFF - manual control")
        elseif not paused and not Isaac.GetPlayer(0):IsDead() and game:GetNumPlayers() == 1 then
            -- A controller can exit and rebind its port while Isaac stays open.
            -- Explicit rearming starts a fresh endpoint and drops queued traffic
            -- from the previous connection; it never restores control by itself.
            if openSocket() then
                resetTransport()
                lastStopReason = nil
                command, sentFrames, lastAccepted = nil, {}, -1
                -- New arming epoch prevents an old reply from the same paused
                -- simulation frame becoming eligible after F8 off/on.
                armCount = armCount + 1
                session = runSession .. ":arm" .. tostring(armCount)
                enabled, status = true, "Jev waiting - manual control"
                clearArmUntil = game:GetRoom():IsClear() and now + CLEAR_ARM_SECONDS or 0
            end
        else status = "Return to gameplay, then F8" end
        sendObservation()
    end
    -- A paused game has no simulation updates. Keep the controller informed.
    if paused and udp and socket.gettime() - lastSentTime >= 0.25 then sendObservation() end
    Isaac.RenderText(fault or status, 50, 30, enabled and 0.4 or 1, 1, 0.5, 1)
end))

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, guarded(function(_, entity, hook, action)
    local direction = actionDirections[action]
    local interaction = interactionActions[action]
    if not direction and not interaction then return nil end
    if not active or not enabled or not command or not udp or game:IsPaused() then return nil end
    if not entity or not entity:ToPlayer() then return nil end
    local player = Isaac.GetPlayer(0)
    if entity.Index ~= player.Index or entity.InitSeed ~= player.InitSeed or player:IsDead() then return nil end
    local frame = game:GetFrameCount()
    local now = socket.gettime()
    if not commandAlive(command, frame, now) then return nil end
    if interaction then
        if command.interaction == "none" then return nil end
        local down = command.interactionPulse == true and command.interaction == interaction
            and frame == command.triggerFrame
        if down and not command.itemInput and (interaction == "active" or interaction == "pocket") then
            local card = interaction == "pocket" and player:GetCard(0) or 0
            command.itemInput = {kind = interaction == "active" and "active" or (card > 0 and "card" or "pill"),
                item = interaction == "active" and player:GetActiveItem(0) or (card > 0 and card or player:GetPill(0))}
        end
        if hook == InputHook.GET_ACTION_VALUE then return down and 1.0 or 0.0 end
        if hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then return down end
        return nil
    end
    local move = appliedMove(command, frame, now, player)
    local down = direction[1] == "move" and moves[move][direction[2]] == true
        or direction[1] == "shoot" and command.shoot == direction[2]
    if hook == InputHook.GET_ACTION_VALUE then return down and 1.0 or 0.0 end
    if hook == InputHook.IS_ACTION_PRESSED then return down end
    if hook == InputHook.IS_ACTION_TRIGGERED then
        return down and frame == command.triggerFrame and command.triggered[action] == true
    end
    return nil
end))

local function onExit()
    disarm("Jev OFF")
    active = false
    closeSocket()
end
mod:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, onExit)
mod:AddCallback(ModCallbacks.MC_POST_GAME_END, onExit)
