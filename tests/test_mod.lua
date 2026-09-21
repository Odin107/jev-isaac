-- Run from the project root: lua tests/test_mod.lua
-- Or pass the main.lua path as arg[1]. Tests mock game/UDP APIs, not the game engine.
local modPath = (arg and arg[1]) or "mod/jev_bridge/main.lua"
local count = 0
local callbacks, mock, player, room, game, udp

local function encode(value)
    if type(value) == "string" then return string.format("%q", value) end
    if type(value) ~= "table" then return tostring(value) end
    local parts = {}
    for key, item in pairs(value) do parts[#parts + 1] = encode(key) .. ":" .. encode(item) end
    return "{" .. table.concat(parts, ",") .. "}"
end

-- Only the action schema is decoded in this harness. Actual game JSON integration
-- remains an in-game check; malformed types are still exercised below.
local function decode(payload)
    if payload == "{malformed}" then error("bad JSON") end
    local result = {}
    for key, raw in payload:gmatch('"([^"]+)":%s*([^,}]+)') do
        if raw:match('^".*"$') then result[key] = raw:sub(2, -2)
        elseif raw == "true" then result[key] = true
        elseif raw == "false" then result[key] = false
        else result[key] = tonumber(raw) end
    end
    return result
end

local function reset(noSocket)
    _G.GetPtrHash = nil
    callbacks = {}
    mock = {frame = 0, now = 1000, paused = false, clear = false, dead = false,
        keys = false, sent = {}, incoming = {}, socketMissing = noSocket, index = 10, logs = {},
        stage = 1, stageType = 0, seed = 456, doors = {}, roomType = 1, roomListIndex = 3,
        socketOpens = 0, socketCloses = 0, sockets = {}}
    udp = nil
    player = {Index = 1, InitSeed = 4, ControllerIndex = 0, Position = {X = 320, Y = 280}, Velocity = {X = 0, Y = 0},
        Type = 1, Variant = 0, SubType = 0, Size = 12, MoveSpeed = 1,
        IsDead = function() return mock.dead end, ToPlayer = function(self) return self end,
        GetHearts = function() return 6 end, GetSoulHearts = function() return 0 end,
        GetMaxHearts = function() return 6 end,
        GetNumCoins = function() return mock.coins or 0 end,
        GetNumBombs = function() return mock.bombs or 0 end,
        GetNumGigaBombs = function() return mock.gigaBombs or 0 end,
        GetBombFlags = function() return mock.bombFlags or 0 end,
        HasTrinket = function(_, id)
            assert(id == 73 or id == 133 or id == 151)
            return (mock.trinketModifiers or {})[id] == true
        end,
        GetPlayerType = function() return mock.playerType or 0 end,
        GetNumKeys = function() return mock.playerKeys or 0 end,
        CanPickRedHearts = function() return mock.canPickRed == true end,
        CanPickSoulHearts = function() return mock.canPickSoul ~= false end,
        CanPickBlackHearts = function() return mock.canPickBlack ~= false end,
        CanPickupItem = function() return mock.canPickupItems ~= false end,
        GetActiveItem = function(_, slot) assert(slot == 0); return mock.activeItem or 0 end,
        GetActiveCharge = function(_, slot) assert(slot == 0); return mock.activeCharge or 0 end,
        GetCard = function(_, slot) assert(slot == 0); return mock.card or 0 end,
        GetPill = function(_, slot) assert(slot == 0); return mock.pill or 0 end,
        GetTrinket = function(_, slot)
            assert(slot == 0 or slot == 1)
            return slot == 0 and (mock.trinket or 0) or (mock.trinket1 or 0)
        end,
        GetCollectibleCount = function() return mock.collectibleCount or 0 end,
        GetCollectibleNum = function(_, id, trueItems)
            assert(trueItems == true)
            mock.inventoryQueries = (mock.inventoryQueries or 0) + 1
            return (mock.inventory or {})[id] or 0
        end,
        HasWeaponType = function(_, weapon) return (mock.weaponTypes or {[1] = true})[weapon] == true end}
    room = {
        GetGridSize = function() return 0 end,
        GetTopLeftPos = function() return {X = 0, Y = 0} end,
        GetBottomRightPos = function() return {X = 640, Y = 560} end,
        IsClear = function() return mock.clear end,
        GetRoomShape = function() return 1 end,
        GetType = function() return mock.roomType end,
        GetDoor = function(_, slot) return mock.doors[slot] end,
        GetFrameCount = function() return mock.frame end,
    }
    game = {
        GetFrameCount = function() return mock.frame end,
        IsPaused = function() return mock.paused end,
        GetNumPlayers = function() return 1 end,
        GetSeeds = function() return {GetStartSeed = function() return 123 end} end,
        GetRoom = function() return room end,
        GetItemPool = function() return {
            IsPillIdentified = function(_, color) return (mock.identifiedPills or {})[color] == true end,
            GetPillEffect = function(_, color, targetPlayer)
                assert(targetPlayer == player)
                assert((mock.identifiedPills or {})[color], "Unknown pill effect must remain hidden")
                mock.pillEffectQueries = (mock.pillEffectQueries or 0) + 1
                return (mock.pillEffects or {})[color] or 0
            end,
        } end,
        GetLevel = function() return {
            GetStage = function() return mock.stage end, GetStageType = function() return mock.stageType end,
            GetDungeonPlacementSeed = function() return mock.seed end,
            GetCurrentRoomDesc = function() return {ListIndex = mock.roomListIndex} end,
            GetCurrentRoomIndex = function() return mock.index end,
        } end,
    }
    local function newUdp()
        -- Each real UDP socket owns its own receive queue. Keep the convenient
        -- mock.incoming alias for existing tests, but never transplant queued
        -- replies from a replaced endpoint into a newly armed connection.
        if udp then udp.incoming = mock.incoming end
        mock.socketOpens = mock.socketOpens + 1
        local candidate = {generation = mock.socketOpens, incoming = {}, closed = false}
        candidate.settimeout = function(_, timeout) assert(timeout == 0); return 1 end
        candidate.setsockname = function(_, host, port)
            assert(host == "127.0.0.1" and port == 0)
            if mock.bindError then return nil, mock.bindError end
            return 1
        end
        candidate.setpeername = function(_, host, port)
            assert(host == "127.0.0.1" and port == 42421); return 1
        end
        candidate.close = function(self)
            if not self.closed then mock.socketCloses = mock.socketCloses + 1 end
            self.closed = true
            mock.closed = true
        end
        candidate.send = function(_, payload) mock.sent[#mock.sent + 1] = payload; return #payload end
        candidate.receive = function(self, size)
            assert(not self.closed, "Cannot receive through a closed endpoint")
            mock.receives = (mock.receives or 0) + 1
            if mock.receiveError then return nil, mock.receiveError end
            local incoming = self == udp and mock.incoming or self.incoming
            if #incoming == 0 then return nil, "timeout" end
            return table.remove(incoming, 1):sub(1, size)
        end
        udp = candidate
        mock.incoming = candidate.incoming
        mock.sockets[#mock.sockets + 1] = candidate
        return candidate
    end
    _G.RegisterMod = function()
        return {AddCallback = function(_, id, fn) callbacks[id] = fn end}
    end
    _G.Game = function() return game end
    _G.Isaac = {GetPlayer = function() return player end, GetRoomEntities = function() return mock.entities or {} end,
        GetItemConfig = function() return {GetCollectible = function(_, subtype)
            assert(subtype > 0)
            return (mock.collectibles or {})[subtype]
        end,
            GetCollectibles = function() return {Size = mock.collectibleSize or 734} end,
            GetCard = function(_, id) return (mock.cards or {})[id] end,
            GetPillEffect = function(_, id) return (mock.pillConfigs or {})[id] end,
        } end,
        RenderText = function(text) mock.hud = text end,
        DebugString = function(text) mock.logs[#mock.logs + 1] = text end}
    _G.Input = {IsButtonTriggered = function(key)
        if key == Keyboard.KEY_F8 then local pressed = mock.keys; mock.keys = false; return pressed end
        if key == mock.pauseKey then mock.pauseKey = nil; return true end
        return false
    end, IsActionTriggered = function(action, controller)
        assert(action == ButtonAction.ACTION_PAUSE and controller == player.ControllerIndex)
        local pressed = mock.pauseAction == true
        mock.pauseAction = false
        return pressed
    end}
    _G.Keyboard = {KEY_F8 = 297, KEY_ESCAPE = 256, KEY_P = 80}
    _G.InputHook = {IS_ACTION_PRESSED = 0, IS_ACTION_TRIGGERED = 1, GET_ACTION_VALUE = 2}
    _G.ButtonAction = {ACTION_LEFT = 0, ACTION_RIGHT = 1, ACTION_UP = 2, ACTION_DOWN = 3,
        ACTION_SHOOTLEFT = 4, ACTION_SHOOTRIGHT = 5, ACTION_SHOOTUP = 6, ACTION_SHOOTDOWN = 7,
        ACTION_BOMB = 8, ACTION_ITEM = 9, ACTION_PILLCARD = 10, ACTION_PAUSE = 12}
    _G.EntityType = {ENTITY_PICKUP = 5, ENTITY_PROJECTILE = 9, ENTITY_BOMBDROP = 4,
        ENTITY_LASER = 7, ENTITY_FIREPLACE = 33, ENTITY_MOVABLE_TNT = 292, ENTITY_EFFECT = 1000}
    _G.PickupVariant = {PICKUP_COLLECTIBLE = 100, PICKUP_TAROTCARD = 300, PICKUP_PILL = 70}
    _G.EntityFlag = {FLAG_FRIENDLY = 1, FLAG_CHARM = 2}
    _G.GridCollisionClass = {COLLISION_NONE = 0}
    _G.GridEntityType = {GRID_SPIKES = 8, GRID_SPIKES_ONOFF = 9,
        GRID_TRAPDOOR = 17, GRID_STAIRS = 18, GRID_TELEPORTER = 23}
    _G.ModCallbacks = {}
    for _, name in ipairs({"MC_POST_GAME_STARTED", "MC_POST_NEW_ROOM", "MC_POST_UPDATE",
        "MC_POST_RENDER", "MC_INPUT_ACTION", "MC_PRE_GAME_EXIT", "MC_POST_GAME_END",
        "MC_USE_ITEM", "MC_USE_CARD", "MC_USE_PILL"}) do ModCallbacks[name] = name end
    _G.require = function(name)
        if name == "json" then return {encode = encode, decode = decode} end
        if mock.socketMissing then error("socket unavailable") end
        assert(name == "socket")
        return {udp = newUdp, gettime = function() return mock.now end}
    end
    dofile(modPath)
    callbacks.MC_POST_GAME_STARTED(nil, false)
end

local function render() callbacks.MC_POST_RENDER() end
local function toggle() mock.keys = true; render() end
local function tick(frame, elapsed)
    mock.now = mock.now + (elapsed or math.max(0, frame - mock.frame) / 30)
    mock.frame = frame
    callbacks.MC_POST_UPDATE()
end
local function input(action, hook, entity)
    return callbacks.MC_INPUT_ACTION(nil, entity or player, hook or 2, action or 0)
end
local function action(changes)
    local latest = mock.sent[#mock.sent]
    local packet = {protocol = 1, type = "action", session = latest:match('"session":"([^"]+)"'),
        room_id = latest:match('"room_id":"([^"]+)"'), frame = mock.frame,
        move = "left", shoot = "right", hold_frames = 6}
    for key, value in pairs(changes or {}) do packet[key] = value end
    return encode(packet)
end
local function queue(changes) mock.incoming[#mock.incoming + 1] = action(changes) end
local function transportState()
    local payload = mock.sent[#mock.sent]
    return decode(assert(payload:match('"transport":(%b{})')))
end
local function check(name, fn)
    local ok, err = pcall(fn)
    assert(ok, name .. ": " .. tostring(err))
    count = count + 1
    print("PASS " .. name)
end

local function pickupEntity(index, variant, subtype, changes)
    local entity = {Index = index, InitSeed = 900 + index, Position = {X = 150, Y = 250},
        Velocity = {X = 1, Y = -2}, Type = EntityType.ENTITY_PICKUP,
        Variant = variant, SubType = subtype, Size = 10, Visible = true,
        Price = 0, OptionsPickupIndex = 0, Wait = 0,
        Exists = function(self) return not self.removed end,
        IsDead = function(self) return self.dead == true end,
        ToNPC = function() return nil end,
        ToPickup = function(self) return self end,
        IsShopItem = function(self) return self.shop == true end}
    for key, value in pairs(changes or {}) do entity[key] = value end
    return entity
end

local function observedPickups()
    local payload = mock.sent[#mock.sent]
    local items = {}
    for item in assert(payload:match('"pickups":(%b[])')):gmatch("%b{}") do
        items[#items + 1] = decode(item)
    end
    return items
end

local function observedItems(name)
    local payload = mock.sent[#mock.sent]
    local items = {}
    for item in assert(payload:match('"' .. name .. '":(%b[])')):gmatch("%b{}") do
        items[#items + 1] = decode(item)
    end
    return items
end

local function npcEntity(index, kind, keepsDoorsClosed)
    return pickupEntity(index, 0, 0, {Type = kind, HitPoints = 100, MaxHitPoints = 100,
        ToNPC = function() return {CanShutDoors = not keepsDoorsClosed} end,
        IsActiveEnemy = function() return true end,
        HasEntityFlags = function() return false end,
        IsVulnerableEnemy = function() return false end,
        CanShutDoors = function() return keepsDoorsClosed end})
end

check("disabled startup and JSON arrays", function()
    reset()
    assert(input() == nil)
    assert(mock.sent[1]:find('"enabled":false', 1, true))
    for _, name in ipairs({"enemies", "projectiles", "hazards", "doors", "pickups"}) do
        assert(mock.sent[1]:find('"' .. name .. '":[]', 1, true))
    end
end)
check("pickup metadata preserves identity geometry price and option constraints", function()
    reset()
    mock.entities = {pickupEntity(21, 20, 2, {Price = -2, shop = true,
        OptionsPickupIndex = 17, Wait = 20})}
    tick(1)
    local item = observedPickups()[1]
    assert(item.id == "921:21" and item.type == 5 and item.variant == 20 and item.subtype == 2)
    assert(item.x == 150 and item.y == 250 and item.vx == 1 and item.vy == -2 and item.radius == 10)
    assert(item.price == -2 and item.shop_item == true and item.options_index == 17 and item.wait == 20)
    assert(item.collectible_kind == 0)
    assert(mock.sent[#mock.sent]:find('"pickup_collection":1', 1, true))
end)
check("collectible kinds come from item config only for existing pedestals", function()
    reset()
    mock.collectibles = {[1] = {Type = 1}, [2] = {Type = 2}, [3] = {Type = 3}}
    mock.entities = {pickupEntity(1, 100, 1), pickupEntity(2, 100, 2), pickupEntity(3, 100, 3),
        pickupEntity(4, 100, 9999), pickupEntity(5, 100, 0), pickupEntity(6, 20, 1)}
    tick(1)
    local items = observedPickups()
    assert(#items == 6)
    for i = 1, 3 do assert(items[i].collectible_kind == i) end
    for i = 4, 6 do assert(items[i].collectible_kind == 0) end
end)
check("pickup observations exclude hidden removed and dead entities", function()
    reset()
    mock.entities = {pickupEntity(1, 10, 1), pickupEntity(2, 10, 1, {Visible = false}),
        pickupEntity(3, 10, 1, {removed = true}), pickupEntity(4, 10, 1, {dead = true})}
    tick(1)
    local items = observedPickups()
    assert(#items == 1 and items[1].id == "901:1")
    mock.entities = {}
    tick(2)
    assert(mock.sent[#mock.sent]:find('"pickups":[]', 1, true))
end)
check("pickup visibility filtering does not remove invisible bomb hazards", function()
    reset()
    mock.entities = {pickupEntity(1, 0, 0, {Type = EntityType.ENTITY_BOMBDROP, Visible = false})}
    tick(1)
    local payload = mock.sent[#mock.sent]
    local hazard = decode(assert(assert(payload:match('"hazards":(%b[])')):match("%b{}")))
    assert(hazard.kind == "bomb" and hazard.type == 4 and hazard.id == "901:1")
    assert(#observedPickups() == 0)
end)
check("shop flags are observed independently from a zero price", function()
    reset()
    mock.entities = {pickupEntity(1, 100, 1, {shop = true}), pickupEntity(2, 100, 1)}
    tick(1)
    local items = observedPickups()
    assert(items[1].price == 0 and items[1].shop_item == true)
    assert(items[2].price == 0 and items[2].shop_item == false)
end)
check("pickup capacity is exactly sixty-four and propagates truncation", function()
    reset()
    mock.entities = {}
    for i = 1, 64 do mock.entities[i] = pickupEntity(i, 20, 1) end
    tick(1)
    local payload = mock.sent[#mock.sent]
    assert(#observedPickups() == 64 and payload:find('"truncated":false', 1, true))
    local flags = decode(assert(payload:match('"truncated_arrays":(%b{})')))
    assert(flags.pickups == false)
    mock.entities[65] = pickupEntity(65, 20, 1)
    tick(2)
    payload = mock.sent[#mock.sent]
    assert(#observedPickups() == 64 and payload:find('"truncated":true', 1, true))
    flags = decode(assert(payload:match('"truncated_arrays":(%b{})')))
    assert(flags.pickups == true and flags.enemies == false and flags.projectiles == false and flags.hazards == false)
end)
check("player collection state uses inventory and eligibility APIs", function()
    reset()
    mock.coins, mock.bombs, mock.playerKeys, mock.activeItem = 23, 7, 5, 105
    mock.canPickRed, mock.canPickSoul, mock.canPickBlack, mock.canPickupItems = true, false, true, false
    tick(1)
    local state = decode(assert(mock.sent[#mock.sent]:match('"player":(%b{})')))
    assert(state.coins == 23 and state.bombs == 7 and state.keys == 5 and state.active_item == 105)
    assert(state.can_pick_red_hearts == true and state.can_pick_soul_hearts == false)
    assert(state.can_pick_black_hearts == true and state.can_pickup_items == false)
    mock.canPickRed, mock.canPickSoul, mock.canPickBlack, mock.canPickupItems = false, true, false, true
    mock.activeItem = 0
    tick(2)
    state = decode(assert(mock.sent[#mock.sent]:match('"player":(%b{})')))
    assert(state.can_pick_red_hearts == false and state.can_pick_soul_hearts == true)
    assert(state.can_pick_black_hearts == false and state.can_pickup_items == true and state.active_item == 0)
end)
check("enemy door blocking observes both results from the base Entity method", function()
    reset()
    mock.entities = {npcEntity(1, 218, false), npcEntity(2, 10, true)}
    mock.clear = true
    tick(1)
    local items = {}
    for value in assert(mock.sent[#mock.sent]:match('"enemies":(%b[])')):gmatch("%b{}") do
        items[#items + 1] = decode(value)
    end
    assert(#items == 2)
    assert(items[1].type == 218 and items[1].keeps_doors_closed == false)
    assert(items[2].type == 10 and items[2].keeps_doors_closed == true)
    assert(items[1].vulnerable == false and items[1].hp == 100 and items[1].max_hp == 100)
end)
check("movable TNT remains a geometry hazard distinct from dropped bombs", function()
    reset()
    mock.entities = {npcEntity(1, EntityType.ENTITY_MOVABLE_TNT, false),
        pickupEntity(2, 0, 0, {Type = EntityType.ENTITY_BOMBDROP})}
    tick(1)
    local payload = mock.sent[#mock.sent]
    assert(payload:find('"enemies":[]', 1, true))
    local items = {}
    for value in assert(payload:match('"hazards":(%b[])')):gmatch("%b{}") do
        items[#items + 1] = decode(value)
    end
    assert(#items == 2 and items[1].kind == "tnt" and items[2].kind == "bomb")
    local tnt = items[1]
    assert(tnt.type == 292 and tnt.id == "901:1" and tnt.radius == 10)
    assert(tnt.x == 150 and tnt.y == 250 and tnt.vx == 1 and tnt.vy == -2)
    assert(tnt.hp == nil and tnt.fuse == nil and tnt.active == nil)
end)
check("movable TNT respects removal death and the shared hazard cap", function()
    reset()
    local removed, dead = npcEntity(1, 292, false), npcEntity(2, 292, false)
    removed.removed, dead.dead = true, true
    mock.entities = {removed, dead}
    tick(1)
    assert(mock.sent[#mock.sent]:find('"hazards":[]', 1, true))
    for index = 3, 163 do mock.entities[index] = npcEntity(index, 292, false) end
    tick(2)
    local payload = mock.sent[#mock.sent]
    local _, total = payload:gsub('"kind":"tnt"', '')
    local flags = decode(assert(payload:match('"truncated_arrays":(%b{})')))
    assert(total == 160 and flags.hazards == true and payload:find('"truncated":true', 1, true))
end)
check("movement, firing, numeric and boolean input hooks", function()
    reset(); toggle(); queue(); tick(0)
    assert(input(0, 2) == 1 and input(1, 2) == 0)
    assert(input(5, 0) == true and input(4, 0) == false)
    assert(input(8) == nil and input(9) == nil)
    tick(1); assert(input(0, 1) == true)
    tick(2); assert(input(0, 1) == false)
end)
check("one-frame hold survives until next simulation tick", function()
    reset(); toggle(); queue({hold_frames = 1}); tick(0)
    tick(1); assert(input() == 1)
    tick(2); assert(input() == nil)
end)
check("maximum hold covers fifteen simulation ticks then returns manual control", function()
    reset(); toggle(); queue({hold_frames = 15}); tick(0)
    for frame = 1, 15 do tick(frame); assert(input() == 1 and input(5) == 1) end
    tick(16); assert(input() == nil and input(5) == nil)
end)
check("bounded movement and local goal control advertise their protocol capabilities", function()
    reset()
    assert(mock.sent[1]:find('"movement_pulses":1', 1, true))
    assert(mock.sent[1]:find('"local_goal_control":1', 1, true))
    assert(not mock.sent[1]:find('"control":', 1, true))
end)
check("transport diagnostics distinguish received rejected and applied commands", function()
    reset(); toggle()
    assert(mock.sent[#mock.sent]:find('"transport_diagnostics":1', 1, true))
    local old = action()
    tick(2)
    queue({move = "right"})
    mock.incoming[#mock.incoming + 1] = old
    queue({protocol = 2})
    tick(3)
    local d = transportState()
    assert(d.received == 3 and d.accepted == 1 and d.rejected == 1)
    assert(d.last_rejection == "protocol" and d.last_received_frame == 3 and d.last_accepted_frame == 2)
    assert(input(1) == 1 and input(0) == 0)
    assert(mock.sent[#mock.sent]:find('"status":"Jev ON - F8 stops"', 1, true))
end)
check("transport rejection reasons identify each action guard without raw payloads", function()
    local cases = {
        {reason = "shape", payload = "[]"},
        {reason = "nested", change = {extra = {value = 1}}},
        {reason = "json", payload = "{malformed}"},
        {reason = "protocol", change = {type = "other"}},
        {reason = "identity", change = {session = "secret-should-not-echo"}},
        {reason = "identity", change = {room_id = "wrong"}},
        {reason = "optional_floor", change = {floor_mode = "true"}},
        {reason = "frame_unknown", change = {frame = 0.5}},
        {reason = "frame_future", change = {frame = 100}},
        {reason = "duration", change = {hold_frames = 16}},
        {reason = "movement", change = {move_frames = 6}},
        {reason = "movement", change = {move_frames = 6, move_distance = 25}},
        {reason = "action", change = {move = "teleport"}},
    }
    for _, case in ipairs(cases) do
        reset(); toggle()
        mock.incoming = {case.payload or action(case.change)}
        tick(1)
        local d = transportState()
        assert(d.received == 1 and d.accepted == 0 and d.rejected == 1, case.reason)
        assert(d.last_rejection == case.reason and d.last_received_frame == 1 and d.last_accepted_frame == -1, case.reason)
        assert(input() == nil)
        assert(not mock.sent[#mock.sent]:find("secret-should-not-echo", 1, true))
    end
    reset(); toggle(); tick(2); queue({frame = 1}); tick(3)
    assert(transportState().last_rejection == "frame_unknown")
    reset(); toggle(); local old = action(); tick(34)
    mock.incoming = {old}; tick(35)
    assert(transportState().last_rejection == "frame_old")
    reset(); toggle(); local duplicate = action(); mock.incoming = {duplicate}; tick(1)
    mock.incoming = {duplicate}; tick(2)
    assert(transportState().last_rejection == "frame_duplicate")
    assert(transportState().accepted == 1 and transportState().rejected == 1)
end)
check("transport diagnostics survive stops and reset only on new arms and rooms", function()
    reset(); toggle(); queue({protocol = 2}); tick(1)
    toggle()
    assert(transportState().received == 1 and transportState().last_rejection == "protocol")
    assert(mock.sent[#mock.sent]:find('"status":"Jev OFF - manual control"', 1, true))
    toggle()
    local d = transportState()
    assert(d.received == 0 and d.accepted == 0 and d.rejected == 0)
    assert(d.last_rejection == "none" and d.last_received_frame == -1 and d.last_accepted_frame == -1)
    queue(); tick(2)
    assert(transportState().accepted == 1)
    callbacks.MC_POST_NEW_ROOM()
    d = transportState()
    assert(d.received == 0 and d.accepted == 0 and d.rejected == 0 and d.last_rejection == "none")
end)
check("transport counts inactive datagrams and accepted immediate revocation", function()
    reset(); queue(); tick(1)
    local d = transportState()
    assert(d.received == 1 and d.accepted == 0 and d.rejected == 1 and d.last_rejection == "inactive")
    toggle(); queue({floor_mode = true}); tick(2)
    queue({frame = 1, floor_mode = false, move = "none", shoot = "none", hold_frames = 1}); tick(3)
    d = transportState()
    assert(d.received == 2 and d.accepted == 2 and d.rejected == 0 and d.last_accepted_frame == 1)
    assert(input() == nil)
    assert(mock.sent[#mock.sent]:find('"status":"Controller released control"', 1, true))
end)
check("each simulation frame sends fresh state once without duplicate update or render sends", function()
    reset()
    local sent = #mock.sent
    tick(0); render(); assert(#mock.sent == sent)
    for frame = 1, 30 do
        player.Position.X = 320 + frame
        tick(frame)
        assert(#mock.sent == sent + frame)
        local payload = mock.sent[#mock.sent]
        assert(payload:find('"frame":' .. frame, 1, true))
        assert(payload:find('"x":' .. (320 + frame), 1, true))
        tick(frame, 0); render(); assert(#mock.sent == sent + frame)
    end
end)
check("paused state repeats every 250 ms while simulation frame is frozen", function()
    reset(); mock.now = 0
    local sent = #mock.sent
    mock.paused = true; render()
    assert(#mock.sent == sent + 1)
    assert(mock.sent[#mock.sent]:find('"paused":true', 1, true))
    tick(0, 0); render(); assert(#mock.sent == sent + 1)
    mock.now = 0.249999; render(); assert(#mock.sent == sent + 1)
    mock.now = 0.25; render(); assert(#mock.sent == sent + 2)
    render(); tick(0, 0); assert(#mock.sent == sent + 2)
    mock.now = 0.499999; render(); assert(#mock.sent == sent + 2)
    mock.now = 0.5; render(); assert(#mock.sent == sent + 3)
    mock.paused = false; render(); assert(#mock.sent == sent + 4)
    mock.now = 0.75; render(); assert(#mock.sent == sent + 4)
end)
check("six movement ticks end independently of fifteen shooting ticks", function()
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    for frame = 1, 6 do tick(frame, 0); assert(input() == 1 and input(5) == 1) end
    -- Isolate the simulation limit from the independent wall-clock limit.
    tick(7, 0)
    for actionId = 0, 3 do
        assert(input(actionId) == 0 and input(actionId, 0) == false and input(actionId, 1) == false)
    end
    assert(input(5) == 1 and input(5, 0) == true)
    tick(9, 0)
    local payload = mock.sent[#mock.sent]
    assert(payload:find('"requested_move":"left"', 1, true))
    assert(payload:find('"applied_move":"none"', 1, true))
    assert(payload:find('"shoot":"right"', 1, true))
    assert(payload:find('"source_frame":0', 1, true))
    assert(payload:find('"move_stop_reason":"time_limit"', 1, true))
    tick(15, 0); assert(input() == 0 and input(5) == 1)
    tick(16, 0); assert(input() == nil and input(5) == nil)
end)
check("six-frame movement stops at 200 ms with frozen simulation", function()
    reset(); toggle(); mock.now = 0
    queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    mock.now = 0.199999; assert(input() == 1 and input(5) == 1)
    mock.now = 0.2; assert(input() == 0 and input(5) == 1)
    -- Once stopped, even a backwards clock change cannot restart this movement.
    mock.now = 0.1; assert(input() == 0)
    mock.now = 0.550001; assert(input() == nil and input(5) == nil)
end)
check("distance brake latches at the requested displacement without changing velocity", function()
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    player.Velocity.X = -4
    player.Position.X = 301; assert(input() == 1)
    player.Position.X = 300; assert(input() == 0 and input(5) == 1)
    assert(player.Velocity.X == -4)
    player.Position.X = 320; assert(input() == 0)
    tick(3, 0)
    assert(mock.sent[#mock.sent]:find('"move_stop_reason":"distance_limit"', 1, true))
end)
check("distance brake uses both axes and the receipt position", function()
    reset(); toggle()
    local delayed = action({hold_frames = 15, move = "up_left", move_frames = 6, move_distance = 20})
    player.Position.X, player.Position.Y = 400, 300
    tick(3, 0); mock.incoming = {delayed}; tick(3, 0)
    assert(input() == 1 and input(2) == 1)
    player.Position.X, player.Position.Y = 388, 284 -- sqrt(12^2 + 16^2) = 20
    assert(input() == 0 and input(2) == 0 and input(5) == 1)
end)
check("new direction interrupts a short movement before either limit", function()
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    tick(1, 0); assert(input() == 1)
    tick(3, 0); queue({move = "right", shoot = "up", hold_frames = 15, move_frames = 6, move_distance = 20}); tick(3, 0)
    assert(input() == 0 and input(1) == 1 and input(5) == 0 and input(6) == 1)
    tick(4, 0); assert(input(1, 1) == true and input(6, 1) == true)
end)
check("new same-direction movement restarts after a brake without retriggering held shooting", function()
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    player.Position.X = 300; assert(input() == 0 and input(5) == 1)
    tick(3, 0); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(3, 0)
    assert(input() == 1 and input(5) == 1)
    tick(4, 0); assert(input(0, 1) == true and input(5, 1) == false)
    player.Position.X = 281; assert(input() == 1)
    player.Position.X = 280; assert(input() == 0 and input(5) == 1)
end)
check("bounded movement fields must form a valid pair", function()
    local invalid = {{move_frames = 6}, {move_distance = 20},
        {move_frames = 0, move_distance = 20}, {move_frames = 7, move_distance = 20},
        {move_frames = 1.5, move_distance = 20}, {move_frames = "6", move_distance = 20},
        {move_frames = false, move_distance = 20}, {move_frames = 6, move_distance = 0},
        {move_frames = 6, move_distance = -1}, {move_frames = 6, move_distance = 24.001},
        {move_frames = 6, move_distance = "20"}, {move_frames = 6, move_distance = false},
        {move_frames = 6, move_distance = 20, hold_frames = 5}}
    for _, change in ipairs(invalid) do reset(); toggle(); queue(change); tick(0); assert(input() == nil) end
    for _, raw in ipairs({"1e999", "-1e999", "null"}) do
        reset(); toggle()
        local packet = action({move_frames = 6, move_distance = 20})
        packet = packet:gsub('"move_distance":20', '"move_distance":' .. raw)
        mock.incoming = {packet}; tick(0); assert(input() == nil)
    end
    reset(); toggle()
    local packet = action({move_frames = 6, move_distance = 20})
    packet = packet:gsub('"move_frames":6', '"move_frames":null'):gsub('"move_distance":20', '"move_distance":null')
    mock.incoming = {packet}; tick(0); assert(input() == nil)
end)
check("movement duration and distance endpoints are accepted", function()
    reset(); toggle(); queue({hold_frames = 1, move_frames = 1, move_distance = 0.01}); tick(0)
    tick(1, 0); assert(input() == 1 and input(5) == 1)
    tick(2, 0); assert(input() == nil and input(5) == nil)
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 24}); tick(0)
    player.Position.X = 296.001; assert(input() == 1)
    player.Position.X = 296; assert(input() == 0)
end)
check("no movement request preserves shooting and reports neutral movement", function()
    reset(); toggle(); queue({move = "none", hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    tick(9)
    assert(input() == 0 and input(5) == 1)
    assert(mock.sent[#mock.sent]:find('"applied_move":"none"', 1, true))
    assert(mock.sent[#mock.sent]:find('"move_stop_reason":"none"', 1, true))
end)
check("pause, disarm, death and source expiry override the split movement hold", function()
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    player.Position.X = 300; assert(input() == 0 and input(5) == 1)
    mock.paused = true; assert(input() == nil and input(5) == nil); render()
    assert(not mock.sent[#mock.sent]:find('"control":', 1, true))
    mock.paused = false; render(); assert(input() == nil and input(5) == nil)
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    toggle(); assert(input() == nil and input(5) == nil)
    reset(); toggle(); queue({hold_frames = 15, move_frames = 6, move_distance = 20}); tick(0)
    mock.dead = true; assert(input() == nil and input(5) == nil)
    reset(); toggle(); local old = action({hold_frames = 15, move_frames = 6, move_distance = 20})
    tick(33, 0); mock.incoming = {old}; tick(33, 0)
    assert(input() == 1 and input(5) == 1)
    tick(34, 0); assert(input() == nil and input(5) == nil)
end)
check("359 ms replies replace held actions without a manual gap", function()
    reset(); toggle(); local first = action({hold_frames = 15})
    tick(11, 0.359); mock.incoming = {first}; tick(11)
    assert(input() == 1 and input(5) == 1)
    tick(12, 0.359 / 11); local replacement = action({move = "right", shoot = "up", hold_frames = 15})
    for frame = 13, 22 do tick(frame, 0.359 / 11); assert(input() == 1 and input(5) == 1) end
    mock.incoming = {replacement}; tick(22)
    assert(input() == 0 and input(1) == 1 and input(5) == 0 and input(6) == 1)
    tick(23); assert(input(1, 1) == true and input(6, 1) == true)
    tick(24); assert(input(1, 1) == false and input(6, 1) == false)
    -- The original hold ends at frame 27. Its replacement remains active.
    tick(27); assert(input(1) == 1 and input(6) == 1)
end)
check("continuous same-direction replies do not repeat input triggers", function()
    reset(); toggle(); queue({hold_frames = 15}); tick(0)
    tick(1); assert(input(0, 1) == true and input(5, 1) == true)
    tick(3); queue({hold_frames = 15}); tick(3)
    tick(4); assert(input(0, 1) == false and input(5, 1) == false)
    assert(input() == 1 and input(5) == 1)
end)
check("delayed action hold is clipped to source age thirty-three", function()
    reset(); toggle(); local delayed = action({hold_frames = 15})
    tick(24); mock.incoming = {delayed}; tick(24)
    assert(input() == 1)
    tick(33); assert(input() == 1)
    -- Only ten ticks and 333 ms have elapsed since receipt: source age,
    -- rather than the hold or wall-clock timeout, must release control.
    tick(34); assert(input() == nil)
end)
check("source age thirty-three is accepted but thirty-four is rejected", function()
    reset(); toggle(); local boundary = action({hold_frames = 15})
    tick(33); mock.incoming = {boundary}; tick(33)
    assert(input() == 1)
    tick(34); assert(input() == nil)
    reset(); toggle(); local expired = action({hold_frames = 15})
    tick(34); mock.incoming = {expired}; tick(34)
    assert(input() == nil)
end)
check("550 ms receipt timeout releases control with frozen simulation", function()
    reset(); toggle(); mock.now = 0; queue({hold_frames = 15}); tick(0)
    mock.now = 0.55; assert(input() == 1 and input(5) == 1)
    mock.now = 0.550001
    assert(input() == nil)
    tick(0); render(); assert(mock.hud == "Jev waiting - manual control")
end)
check("malformed, mismatched and invalid actions rejected", function()
    local invalid = {{protocol = 2}, {type = "other"}, {session = "wrong"}, {room_id = "wrong"},
        {hold_frames = 0}, {hold_frames = 16}, {hold_frames = 1.5}, {hold_frames = "3"},
        {frame = 100}, {frame = -1}, {frame = 0.5}, {move = "teleport"}, {shoot = "up_left"}}
    for _, change in ipairs(invalid) do reset(); toggle(); queue(change); tick(0); assert(input() == nil) end
    reset(); toggle(); mock.incoming = {"{malformed}", "[]", string.rep("x", 3000)}; tick(0)
    assert(input() == nil)
end)
check("commands referencing unsent, old, duplicate frames rejected", function()
    -- Skip a frame entirely so the rejected source really was never observed.
    reset(); toggle(); tick(2); queue({frame = 1}); tick(2); assert(input() == nil)
    reset(); toggle(); local old = action(); tick(34); mock.incoming = {old}; tick(34); assert(input() == nil)
    reset(); toggle(); local first = action(); queue(); tick(0)
    tick(7); mock.incoming = {first}; tick(7); assert(input() == nil)
end)
check("newest valid action wins in the bounded receive batch", function()
    reset(); toggle(); local old = action(); tick(3)
    queue({move = "right"}); mock.incoming[#mock.incoming + 1] = old; tick(3)
    assert(input(1) == 1 and input(0) == 0)
    mock.incoming = {}
    for _ = 1, 100 do mock.incoming[#mock.incoming + 1] = "{malformed}" end
    mock.receives = 0; tick(4); assert(mock.receives == 32)
end)
check("disable and pause release all control", function()
    reset(); toggle(); queue({hold_frames = 15}); tick(0); toggle(); assert(input() == nil)
    reset(); toggle(); queue({hold_frames = 15}); tick(0); mock.paused = true
    assert(input() == nil); render()
    assert(mock.sent[#mock.sent]:find('"paused":true', 1, true))
    mock.paused = false; render(); assert(input() == nil)
    assert(mock.sent[#mock.sent]:find('"enabled":false', 1, true))
end)
check("room revisit changes ID and old replies cannot apply", function()
    reset(); toggle(); queue({hold_frames = 15}); tick(0); assert(input() == 1)
    local old = action(); local before = mock.sent[#mock.sent]:match('"room_id":"([^"]+)"')
    callbacks.MC_POST_NEW_ROOM()
    assert(input() == nil)
    local after = mock.sent[#mock.sent]:match('"room_id":"([^"]+)"')
    assert(before ~= after)
    toggle(); mock.incoming = {old}; tick(0); assert(input() == nil)
end)
check("same-frame rearm uses a new session epoch", function()
    reset(); toggle(); local old = action()
    local before = mock.sent[#mock.sent]:match('"session":"([^"]+)"')
    toggle(); toggle()
    local after = mock.sent[#mock.sent]:match('"session":"([^"]+)"')
    assert(before ~= after)
    mock.incoming = {old}; tick(0); assert(input() == nil)
    queue(); tick(0); assert(input() == 1)
end)
check("entity cap and truncation flags keep datagrams bounded", function()
    reset()
    mock.entities = {}
    for i = 1, 100 do
        mock.entities[i] = {Index = i, InitSeed = i, Position = {X = i, Y = i},
            Velocity = {X = 0, Y = 0}, Type = 9, Variant = 0, SubType = 0, Size = 4,
            Exists = function() return true end, IsDead = function() return false end,
            ToNPC = function() return nil end}
    end
    tick(3)
    local payload = mock.sent[#mock.sent]
    assert(payload:find('"truncated":true', 1, true))
    local _, total = payload:gsub('"radius":4', '')
    assert(total == 96 and #payload < 60000)
end)
check("room clear, death and exit stop control", function()
    reset(); toggle(); queue({hold_frames = 15}); tick(0); mock.clear = true; tick(1); assert(input() == nil)
    reset(); toggle(); queue({hold_frames = 15}); tick(0); mock.dead = true; assert(input() == nil)
    reset(); toggle(); queue({hold_frames = 15}); tick(0); callbacks.MC_PRE_GAME_EXIT(); assert(input() == nil and mock.closed)
end)
check("no network dependency and receive failure remain manual", function()
    reset(true); render(); assert(input() == nil and mock.hud:find("luadebug", 1, true))
    reset(); toggle(); queue(); tick(0); mock.receiveError = "connection refused"; tick(1)
    assert(input() == nil and mock.closed)
end)
check("other entities cannot inherit player commands", function()
    reset(); toggle(); queue(); tick(0)
    assert(input(0, 2, {ToPlayer = function() return nil end}) == nil)
    assert(input(0, 2, {ToPlayer = function(self) return self end, Index = 2, InitSeed = 5}) == nil)
end)
local function addDoor(slot, target, x, y, opened, locked, kind)
    mock.doors[slot] = {TargetRoomIndex = target, TargetRoomType = kind or 1,
        Position = {X = x, Y = y}, IsOpen = function() return opened end,
        IsLocked = function() return locked end}
end
local function observationHas(fragment)
    return mock.sent[#mock.sent]:find(fragment, 1, true) ~= nil
end
check("manual F8 arms replace the UDP endpoint but F8 off does not", function()
    reset()
    local startup = udp
    assert(mock.socketOpens == 1 and mock.socketCloses == 0)
    assert(transportState().socket_epoch == 1)
    toggle()
    local armed = udp
    assert(armed ~= startup and startup.closed)
    assert(mock.socketOpens == 2 and mock.socketCloses == 1)
    assert(transportState().socket_epoch == 2 and observationHas('"enabled":true'))
    toggle()
    assert(udp == armed and not armed.closed)
    assert(mock.socketOpens == 2 and mock.socketCloses == 1)
    assert(transportState().socket_epoch == 2 and observationHas('"enabled":false'))
    toggle()
    assert(udp ~= armed and armed.closed)
    assert(mock.socketOpens == 3 and mock.socketCloses == 2)
    assert(transportState().socket_epoch == 3 and observationHas('"enabled":true'))
end)
check("replacement discards the old socket queue and authenticates late old replies", function()
    reset(); toggle()
    local oldEndpoint, oldReply = udp, action({floor_mode = true})
    mock.incoming[#mock.incoming + 1] = oldReply
    toggle(); toggle()
    assert(oldEndpoint.closed and #oldEndpoint.incoming == 1)
    assert(udp ~= oldEndpoint and #mock.incoming == 0)
    -- Delayed delivery to the retired endpoint cannot appear on the new one.
    oldEndpoint.incoming[#oldEndpoint.incoming + 1] = oldReply
    tick(1)
    assert(transportState().received == 0 and input() == nil)
    -- Even an old reply delivered to the current endpoint fails the arm epoch.
    mock.incoming[#mock.incoming + 1] = oldReply
    tick(2)
    assert(transportState().last_rejection == "identity" and input() == nil)
    queue({floor_mode = true}); tick(3)
    assert(transportState().accepted == 1 and input() == 1)
    assert(observationHas('"floor_mode":true'))
end)
check("failed F8 socket replacement stays disarmed until a fresh successful arm", function()
    reset(); toggle(); queue({floor_mode = true}); tick(0)
    local oldReply, oldEpoch = action({floor_mode = true}), transportState().socket_epoch
    toggle()
    mock.bindError = "mock bind failed"
    toggle()
    assert(input() == nil and udp.closed)
    assert(mock.socketOpens == 3 and mock.socketCloses == 3)
    assert(mock.hud:find("loopback setup failed", 1, true))
    mock.bindError = nil
    toggle()
    assert(input() == nil and observationHas('"enabled":true'))
    assert(transportState().socket_epoch == oldEpoch + 1)
    mock.incoming = {oldReply}; tick(1)
    assert(input() == nil and transportState().last_rejection == "identity")
    queue({floor_mode = true}); tick(2); assert(input() == 1)
end)
check("paused and dead players cannot reopen a socket with F8", function()
    for _, condition in ipairs({"paused", "dead"}) do
        reset(); mock[condition] = true; toggle()
        assert(mock.socketOpens == 1 and mock.socketCloses == 0, condition)
        assert(input() == nil and observationHas('"enabled":false'), condition)
    end
end)
check("receive polling diagnostics separate empty reads from received packets", function()
    reset(); toggle(); tick(1)
    local d = transportState()
    assert(d.receive_polls == 1 and d.receive_timeouts == 1 and d.received == 0)
    queue(); tick(2)
    d = transportState()
    assert(d.receive_polls == 3 and d.receive_timeouts == 2 and d.received == 1)
    assert(d.accepted == 1 and d.rejected == 0)
    mock.receiveError = "mock receive failure"; tick(3); render()
    assert(input() == nil and udp.closed)
    mock.receiveError = nil; toggle()
    d = transportState()
    assert(d.receive_polls == 0 and d.receive_timeouts == 0 and d.received == 0)
end)
check("authorized room transitions preserve the endpoint but replace room authentication", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    local endpoint, epoch = udp, transportState().socket_epoch
    local oldReply = action({floor_mode = true, move = "right"})
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    assert(udp == endpoint and not endpoint.closed)
    assert(mock.socketOpens == 2 and mock.socketCloses == 1)
    assert(transportState().socket_epoch == epoch and observationHas('"enabled":true'))
    mock.incoming = {oldReply}; tick(1)
    assert(input() == nil and transportState().last_rejection == "identity")
    queue({floor_mode = true, move = "right"}); tick(2)
    assert(input(1) == 1 and transportState().accepted == 1)
    assert(udp == endpoint and transportState().socket_epoch == epoch)
end)
check("clear-room timeouts identify missing or expired controller permission", function()
    reset(); mock.clear = true; toggle(); tick(1, 2.001)
    assert(observationHas('"status":"No valid controller reply: F8 retries"'))
    assert(observationHas('"enabled":false') and input() == nil)
    reset(); mock.clear = true; toggle(); queue({floor_mode = true}); tick(0)
    tick(1, 2.001)
    assert(observationHas('"status":"Controller timed out: F8 retries"'))
    assert(observationHas('"enabled":false') and input() == nil)
    -- Room-only mode still stops for the actual successful room clear.
    reset(); toggle(); queue(); tick(0); mock.clear = true; tick(1)
    assert(observationHas('"status":"Room cleared: Jev stopped"'))
    assert(observationHas('"enabled":false') and input() == nil)
end)
check("floor observation includes only the room doors and stable floor identity", function()
    reset(); addDoor(2, 11, 640, 280, true, false, 5); tick(1)
    assert(observationHas('"floor_control":1') and observationHas('"id":"1:0:456"'))
    assert(observationHas('"room_index":10') and observationHas('"target_index":11'))
    assert(observationHas('"room_list_index":3'))
    assert(observationHas('"target_type":5') and observationHas('"locked":false'))
    assert(observationHas('"open":true') and observationHas('"doors":[{'))
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    assert(observationHas('"id":"1:0:456"') and observationHas('"room_index":11'))
    assert(observationHas('"room_list_index":3'))
    mock.seed = 457; callbacks.MC_POST_NEW_ROOM()
    assert(observationHas('"id":"1:0:457"'))
end)
check("trapdoors stairs and teleporters are observed even without grid collision", function()
    reset()
    room.GetGridSize = function() return 4 end
    room.GetGridEntity = function(_, index)
        return {GetType = function() return ({17, 18, 23, 0})[index + 1] end, State = 1}
    end
    room.GetGridCollision = function() return 0 end
    room.GetGridPosition = function(_, index) return {X = 80 + index * 40, Y = 100} end
    tick(1)
    assert(observationHas('"type":17') and observationHas('"type":18') and observationHas('"type":23'))
    assert(not observationHas('"index":3'))
end)
check("clear-room F8 provides a bounded chance to accept floor authorization", function()
    reset(); mock.clear = true; toggle(); tick(1)
    assert(observationHas('"enabled":true'))
    queue({floor_mode = true}); tick(1); tick(2)
    assert(observationHas('"floor_mode":true') and observationHas('"enabled":true'))
    reset(); mock.clear = true; toggle(); tick(1, 2.001)
    assert(observationHas('"enabled":false') and input() == nil)
    reset(); mock.clear = true; toggle(); queue(); tick(0); tick(1)
    assert(observationHas('"enabled":false'))
end)
check("floor authorization preserves room clear and expires without a controller", function()
    reset(); toggle(); queue({floor_mode = true, hold_frames = 15}); tick(0)
    mock.clear = true; tick(1)
    assert(input() == 1 and observationHas('"enabled":true'))
    tick(2, 2); assert(input() == nil and observationHas('"enabled":false'))
end)
check("room transition preserves recent floor permission but never old controls", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right", hold_frames = 15}); tick(0)
    local old = action({floor_mode = true, hold_frames = 15})
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    assert(input() == nil and observationHas('"enabled":true'))
    mock.incoming = {old}; tick(0); assert(input() == nil)
    queue({floor_mode = true, move = "right"}); tick(0)
    assert(input() == 0 and input(1) == 1)
    reset(); toggle(); queue({floor_mode = true}); tick(0)
    mock.now = mock.now + 2; mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    assert(observationHas('"enabled":false') and input() == nil)
end)
check("authorized approach survives transition pause only into the expected room", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.paused = true; render()
    assert(observationHas('"floor_transition":true') and observationHas('"enabled":true'))
    assert(input() == nil)
    mock.index = 11; callbacks.MC_POST_NEW_ROOM(); mock.paused = false; render()
    assert(observationHas('"enabled":true') and observationHas('"floor_transition":false'))
    queue({floor_mode = true}); tick(0); assert(input() == 1)
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.paused = true; render(); mock.index = 12; callbacks.MC_POST_NEW_ROOM()
    assert(observationHas('"enabled":false'))
end)
check("new-room-before-pause callback ordering also clears and resumes safely", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    mock.paused = true; render(); assert(input() == nil and observationHas('"enabled":true'))
    mock.paused = false; render(); assert(observationHas('"enabled":true'))
    queue({floor_mode = true}); tick(0); assert(input() == 1)
end)
check("new-room controls before the unpause render do not discard transition evidence", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.paused = true; render(); mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    mock.paused = false; tick(1); queue({floor_mode = true}); tick(1)
    assert(input() == 1)
    render(); assert(input() == 1 and observationHas('"enabled":true'))
end)
check("new-room controls without a pause edge refresh permission beyond the old lease", function()
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    queue({floor_mode = true}); tick(0); render()
    for frame = 1, 90 do
        tick(frame); queue({floor_mode = true}); tick(frame, 0); render()
        assert(input() == 1 and observationHas('"enabled":true'))
    end
    assert(observationHas('"floor_transition":false'))
end)
check("unanticipated room changes cannot carry floor control", function()
    reset(); toggle(); queue({floor_mode = true}); tick(0)
    mock.index = 11; callbacks.MC_POST_NEW_ROOM()
    assert(input() == nil and observationHas('"enabled":false'))
    for _, setup in ipairs({"closed", "locked", "away", "far"}) do
        reset(); mock.clear = true
        addDoor(2, 11, 640, 280, setup ~= "closed", setup == "locked")
        player.Position.X = setup == "far" and 320 or 610
        toggle(); queue({floor_mode = true, move = setup == "away" and "left" or "right"}); tick(0)
        mock.paused = true; render()
        assert(observationHas('"enabled":false'), setup)
    end
end)
check("ordinary pauses and canceled doorway pauses revoke floor permission", function()
    reset(); toggle(); queue({floor_mode = true}); tick(0)
    mock.paused = true; render(); assert(observationHas('"enabled":false'))
    mock.paused = false; render(); assert(observationHas('"enabled":false'))
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
    player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
    mock.paused = true; render(); mock.paused = false; render()
    assert(observationHas('"enabled":false') and observationHas('"floor_mode":false'))
end)
check("pause keys, F8, death and new floors revoke pending floor transitions", function()
    for _, stop in ipairs({"escape", "p", "f8", "dead", "stage", "seed", "timeout"}) do
        reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false)
        player.Position.X = 610; toggle(); queue({floor_mode = true, move = "right"}); tick(0)
        mock.paused = true; render()
        if stop == "escape" then mock.pauseKey = Keyboard.KEY_ESCAPE; render()
        elseif stop == "p" then mock.pauseKey = Keyboard.KEY_P; render()
        elseif stop == "f8" then toggle()
        elseif stop == "dead" then mock.dead = true; tick(1)
        elseif stop == "stage" then mock.stage = 2; callbacks.MC_POST_NEW_ROOM()
        elseif stop == "seed" then mock.seed = 457; callbacks.MC_POST_NEW_ROOM()
        else mock.now = mock.now + 2; render() end
        assert(observationHas('"enabled":false') and observationHas('"floor_mode":false'), stop)
        assert(input() == nil)
    end
end)
local function approachBoss(kind)
    reset(); mock.clear = true; addDoor(2, 11, 640, 280, true, false, kind or 5)
    mock.roomFrame = 20
    room.GetFrameCount = function() return mock.roomFrame end
    player.Position.X = 610
    toggle(); queue({floor_mode = true, move = "right", hold_frames = 15}); tick(0)
end
local function enterBoss()
    mock.index, mock.roomType, mock.roomFrame, mock.clear = 11, 5, 0, false
    callbacks.MC_POST_NEW_ROOM()
end
check("boss introduction preserves permission beyond the ordinary lease without keeping old commands", function()
    approachBoss()
    local oldReply = action({floor_mode = true, move = "right"})
    mock.paused = true; render()
    assert(input() == nil and observationHas('"floor_transition":true'))
    mock.now = mock.now + 3; render()
    assert(observationHas('"enabled":true') and input() == nil)
    enterBoss()
    assert(observationHas('"enabled":true') and input() == nil)
    mock.paused = false; render()
    assert(observationHas('"enabled":true') and observationHas('"floor_transition":true'))
    mock.incoming = {oldReply}; tick(1)
    assert(input() == nil and transportState().last_rejection == "identity")
    mock.roomFrame = 3; tick(3)
    queue({floor_mode = true, move = "left"}); tick(3); tick(4)
    assert(input() == 1 and observationHas('"floor_transition":false'))
end)
check("boss permission handles new-room-first callbacks and an early double pause", function()
    approachBoss()
    enterBoss()
    assert(observationHas('"enabled":true') and input() == nil)
    -- An early unpaused update may receive a command before the intro render.
    tick(1); queue({floor_mode = true}); tick(1)
    assert(observationHas('"floor_transition":true'))
    mock.paused = true; render()
    assert(input() == nil and observationHas('"enabled":true'))
    mock.now = mock.now + 3; mock.roomFrame = 1
    mock.paused = false; render(); tick(2)
    queue({floor_mode = true}); tick(2)
    assert(observationHas('"floor_transition":true'))
    mock.paused = true; render()
    assert(input() == nil and observationHas('"enabled":true'))
    mock.paused = false; mock.roomFrame = 3; render(); tick(3)
    queue({floor_mode = true}); tick(3); tick(4)
    assert(input() == 1 and observationHas('"floor_transition":false'))
end)
check("boss commands arriving before unpause render retain transition evidence", function()
    approachBoss(); mock.paused = true; render()
    mock.now = mock.now + 3; enterBoss()
    mock.roomFrame = 3; mock.paused = false; tick(1)
    queue({floor_mode = true}); tick(1)
    assert(observationHas('"enabled":true') and observationHas('"floor_transition":true'))
    render()
    assert(observationHas('"enabled":true'))
    tick(2); queue({floor_mode = true}); tick(2); tick(3)
    assert(input() == 1 and observationHas('"floor_transition":false'))
end)
check("boss introduction deadline is fixed and never renewed by arrival or renders", function()
    approachBoss()
    local started = mock.now
    mock.paused = true; render()
    mock.now = started + 6; render(); enterBoss()
    assert(observationHas('"enabled":true') and input() == nil)
    for _, elapsed in ipairs({6.5, 7, 7.9}) do
        mock.now = started + elapsed; render()
        assert(observationHas('"enabled":true') and input() == nil)
    end
    mock.now = started + 8.001; render()
    assert(observationHas('"enabled":false') and observationHas('"floor_transition":false'))
    assert(input() == nil)
end)
check("boss arrival with a missed pause callback requires frozen frames and the fixed deadline", function()
    for _, case in ipairs({{frames = 0, elapsed = 3, allowed = true},
                          {frames = 2, elapsed = 3, allowed = true},
                          {frames = 3, elapsed = 3, allowed = false},
                          {frames = 0, elapsed = 8, allowed = false}}) do
        approachBoss()
        -- No render or update occurs during the intro: new-room is the first
        -- callback that can authenticate the arrival after the ordinary lease.
        mock.now = mock.now + case.elapsed
        mock.frame = case.frames
        enterBoss()
        assert(observationHas('"enabled":' .. tostring(case.allowed)))
        assert(observationHas('"floor_transition":' .. tostring(case.allowed)))
        assert(input() == nil)
    end
end)
check("ordinary and treasure doors retain the original two-second transition limit", function()
    for _, kind in ipairs({1, 4}) do
        approachBoss(kind)
        mock.paused = true; render()
        mock.now = mock.now + 2.001; render()
        assert(observationHas('"enabled":false') and input() == nil, tostring(kind))
    end
end)
check("closed locked distant or inward boss approaches never earn the longer permit", function()
    for _, setup in ipairs({"closed", "locked", "far", "away"}) do
        reset(); mock.clear = true
        addDoor(2, 11, 640, 280, setup ~= "closed", setup == "locked", 5)
        player.Position.X = setup == "far" and 320 or 610
        toggle(); queue({floor_mode = true, move = setup == "away" and "left" or "right"}); tick(0)
        mock.paused = true; render()
        assert(observationHas('"enabled":false') and input() == nil, setup)
    end
end)
check("boss introduction permission remains subject to all explicit stop and identity guards", function()
    for _, stop in ipairs({"escape", "p", "f8", "dead", "stage", "seed", "wrong_target", "wrong_type"}) do
        approachBoss(); mock.paused = true; render()
        mock.now = mock.now + 3; render()
        assert(observationHas('"enabled":true'), stop)
        if stop == "escape" then mock.pauseKey = Keyboard.KEY_ESCAPE; render()
        elseif stop == "p" then mock.pauseKey = Keyboard.KEY_P; render()
        elseif stop == "f8" then toggle()
        elseif stop == "dead" then mock.dead = true; tick(1)
        elseif stop == "stage" then mock.stage = 2; enterBoss()
        elseif stop == "seed" then mock.seed = 457; enterBoss()
        elseif stop == "wrong_type" then
            mock.index, mock.roomType, mock.roomFrame, mock.clear = 11, 1, 0, false
            callbacks.MC_POST_NEW_ROOM()
        else mock.index = 12; callbacks.MC_POST_NEW_ROOM() end
        assert(observationHas('"enabled":false') and observationHas('"floor_transition":false'), stop)
        assert(input() == nil, stop)
    end
end)
check("the first stop reason survives cosmetic unpause updates and resets on a new F8 arm", function()
    approachBoss(); mock.paused = true; render()
    mock.pauseKey = Keyboard.KEY_ESCAPE; render()
    assert(observationHas('"last_stop_reason":"Paused: Jev stopped"'))
    mock.paused = false; render()
    assert(observationHas('"last_stop_reason":"Paused: Jev stopped"'))
    mock.index = 12; callbacks.MC_POST_NEW_ROOM()
    assert(observationHas('"last_stop_reason":"Paused: Jev stopped"'))
    toggle()
    assert(observationHas('"enabled":true'))
    assert(not observationHas('"last_stop_reason":"Paused: Jev stopped"'))
end)
check("floor mode only accepts booleans and absent mode revokes authorization", function()
    for _, value in ipairs({1, 0, "true", "false"}) do
        reset(); toggle(); queue({floor_mode = value}); tick(0); assert(input() == nil)
    end
    reset(); toggle()
    mock.incoming = {action({floor_mode = true}):gsub('"floor_mode":true', '"floor_mode":null')}
    tick(0); assert(input() == nil)
    reset(); toggle(); queue({floor_mode = true}); tick(0)
    tick(1); queue(); tick(1); mock.clear = true; tick(2)
    assert(observationHas('"enabled":false'))
end)
check("same-source neutral floor revocation stops immediately without reopening movement", function()
    reset(); toggle(); queue({floor_mode = true, hold_frames = 15}); tick(0)
    queue({floor_mode = false, move = "none", shoot = "none", hold_frames = 1}); tick(0)
    assert(input() == nil)
    queue({floor_mode = true}); tick(1); assert(input() == nil)
    for _, change in ipairs({{move = "left"}, {shoot = "right"}, {hold_frames = 2}, {floor_mode = true}}) do
        reset(); toggle(); queue({floor_mode = true, hold_frames = 15}); tick(0)
        local packet = {floor_mode = false, move = "none", shoot = "none", hold_frames = 1}
        for key, value in pairs(change) do packet[key] = value end
        queue(packet); tick(0); assert(input() == 1)
    end
end)
check("new-frame neutral release disarms before room clear can hide its cause", function()
    for _, clear in ipairs({false, true}) do
        reset(); mock.clear = clear; toggle()
        queue({floor_mode = true, hold_frames = 15}); tick(0); tick(1)
        queue({frame = 1, floor_mode = false, move = "none", shoot = "none", hold_frames = 1})
        tick(2)
        assert(observationHas('"enabled":false') and input() == nil)
        assert(observationHas('"last_stop_reason":"Controller released control"'))
        tick(3)
        assert(not observationHas('"last_stop_reason":"Room cleared: Jev stopped"'))
    end
end)
check("navigation release reports blocked route and one fresh F8 rearms", function()
    for _, sameFrame in ipairs({false, true}) do
        reset(); mock.clear = true; toggle()
        queue({floor_mode = true, hold_frames = 15}); tick(0)
        if not sameFrame then tick(1) end
        queue({frame = sameFrame and 0 or 1, floor_mode = false, move = "none", shoot = "none",
            hold_frames = 1, stop_reason = "navigation"}); tick(sameFrame and 0 or 2)
        assert(input() == nil)
        if sameFrame then tick(1) end
        assert(observationHas('"enabled":false') and input() == nil)
        assert(observationHas('"status":"Route blocked: F8 retries"'))
        toggle(); assert(observationHas('"enabled":true'))
        queue({floor_mode = true}); tick(sameFrame and 1 or 2)
        assert(input() == 1)
    end
end)
check("release reason cannot authorize motion or bypass identity and frame checks", function()
    for _, change in ipairs({{stop_reason = "unknown"}, {stop_reason = true}, {move = "left"},
        {shoot = "up"}, {floor_mode = true}, {hold_frames = 2}, {interaction = "bomb", interaction_id = "x"},
        {move_frames = 1, move_distance = 1}, {session = "wrong"}, {room_id = "wrong"}, {frame = 99}}) do
        reset(); toggle(); queue({floor_mode = true, hold_frames = 15}); tick(0)
        local packet = {floor_mode = false, move = "none", shoot = "none", hold_frames = 1,
            stop_reason = "navigation"}
        for key, value in pairs(change) do packet[key] = value end
        queue(packet); tick(0)
        assert(input() == 1)
    end
    reset(); toggle(); queue({floor_mode = true, hold_frames = 15}); tick(0)
    mock.incoming = {(action({floor_mode = false, move = "none", shoot = "none", hold_frames = 1,
        stop_reason = "navigation"}):gsub('"stop_reason":"navigation"', '"stop_reason":null'))}
    tick(0); assert(input() == 1)
end)
check("boss transition deadline cannot be renewed by an update before the next render", function()
    for _, expiresDuringReceive in ipairs({false, true}) do
        approachBoss()
        local began = mock.now
        mock.paused = true; render(); enterBoss()
        mock.now = began + 7.99
        mock.roomFrame = 3; mock.paused = false; render(); tick(1, 0)
        queue({floor_mode = true, move = "left"})
        if expiresDuringReceive then
            -- The update starts within the permit, but processing the queued
            -- packet crosses its deadline before acceptance can renew a lease.
            local receive = udp.receive
            udp.receive = function(self, size)
                local payload, err = receive(self, size)
                if payload then mock.now = began + 8.01 end
                return payload, err
            end
            tick(2, .001)
        else
            -- No render occurs at expiry: the update must enforce it first.
            tick(2, .02)
        end
        assert(observationHas('"enabled":false'), tostring(expiresDuringReceive))
        assert(observationHas('"floor_transition":false'))
        assert(observationHas('"last_stop_reason":"Transition expired: F8 enables Jev"'))
        assert(input() == nil and transportState().accepted == 0)
    end
end)
check("interaction packets validate enum identity length and explicit nulls before acceptance", function()
    local invalid = {{interaction = "unknown"}, {interaction = true}, {interaction = 1},
        {interaction = "bomb"}, {interaction = "bomb", interaction_id = ""},
        {interaction = "active", interaction_id = string.rep("x", 129)},
        {interaction = "pocket", interaction_id = 7}}
    for _, change in ipairs(invalid) do
        reset(); toggle(); queue(change); tick(1)
        assert(transportState().last_rejection == "interaction")
        assert(input(ButtonAction.ACTION_BOMB) == nil)
    end
    for _, name in ipairs({"interaction", "interaction_id"}) do
        reset(); toggle()
        local raw = action({interaction = "bomb", interaction_id = "test"})
        raw = raw:gsub('"' .. name .. '":"[^"]+"', '"' .. name .. '":null')
        mock.incoming = {raw}; tick(1)
        assert(transportState().last_rejection == "interaction")
    end
    reset(); toggle(); queue({interaction = "bomb", interaction_id = string.rep("x", 128)}); tick(0)
    tick(1); assert(input(ButtonAction.ACTION_BOMB) == 1)
end)
check("bomb active and pocket interactions pulse one simulation tick with correct hook types", function()
    for _, kind in ipairs({{"bomb", ButtonAction.ACTION_BOMB}, {"active", ButtonAction.ACTION_ITEM},
        {"pocket", ButtonAction.ACTION_PILLCARD}}) do
        reset(); toggle(); queue({interaction = kind[1], interaction_id = "one", hold_frames = 15}); tick(0)
        assert(input(kind[2]) == 0)
        tick(1)
        assert(input(kind[2], InputHook.GET_ACTION_VALUE) == 1)
        assert(input(kind[2], InputHook.IS_ACTION_PRESSED) == true)
        assert(input(kind[2], InputHook.IS_ACTION_TRIGGERED) == true)
        assert(input(kind[2], 999) == nil)
        -- Repeated engine queries are stable within a tick, then release.
        assert(input(kind[2]) == 1)
        tick(2)
        assert(input(kind[2]) == 0 and input(kind[2], InputHook.IS_ACTION_TRIGGERED) == false)
        assert(input(ButtonAction.ACTION_PAUSE) == nil)
    end
end)
check("interaction replay IDs survive rearming and room changes but reset on a new run", function()
    reset(); toggle(); queue({interaction = "bomb", interaction_id = "once"}); tick(0); tick(1)
    assert(input(ButtonAction.ACTION_BOMB) == 1)
    tick(20); queue({interaction = "active", interaction_id = "once"}); tick(20); tick(21)
    assert(input(ButtonAction.ACTION_ITEM) == 0)
    assert(observationHas('"interaction_status":"duplicate"'))
    toggle(); tick(40); toggle(); queue({interaction = "bomb", interaction_id = "once"}); tick(40); tick(41)
    assert(input(ButtonAction.ACTION_BOMB) == 0)
    mock.index = 12; callbacks.MC_POST_NEW_ROOM(); tick(60); toggle()
    queue({interaction = "bomb", interaction_id = "once"}); tick(60); tick(61)
    assert(input(ButtonAction.ACTION_BOMB) == 0)
    callbacks.MC_POST_GAME_STARTED(nil, false); toggle()
    queue({interaction = "bomb", interaction_id = "once"}); tick(61); tick(62)
    assert(input(ButtonAction.ACTION_BOMB) == 1)
end)
local function useItem(kind, clear)
    reset(); mock.clear = clear == true
    mock.activeItem, mock.activeCharge, mock.card, mock.pill = 41, 2, 19, 5
    if kind == "pill" then mock.card = 0 end
    toggle()
    local interaction = kind == "active" and "active" or "pocket"
    queue({interaction = interaction, interaction_id = "use-one", floor_mode = true})
    tick(0); mock.frame = 1
    assert(input(kind == "active" and ButtonAction.ACTION_ITEM or ButtonAction.ACTION_PILLCARD) == 1)
    if kind == "active" then
        if callbacks.MC_USE_ITEM then assert(callbacks.MC_USE_ITEM(nil, 41, {}, player, 4, 0, 0) == nil) end
        mock.activeCharge = 0
    elseif kind == "card" then
        if callbacks.MC_USE_CARD then callbacks.MC_USE_CARD(nil, 19, player, 0) end
        mock.card = 0
    else
        mock.card = 0
        if callbacks.MC_USE_PILL then callbacks.MC_USE_PILL(nil, 7, player, 0) end
        mock.pill = 0
    end
end
check("confirmed item animations preserve control and resume only fresh inputs", function()
    for _, kind in ipairs({"active", "card", "pill"}) do
        for _, clear in ipairs({false, true}) do
            useItem(kind, clear)
            local oldPacket = action({frame = 0, floor_mode = true})
            mock.paused = true; render()
            assert(observationHas('"enabled":true'), kind)
            assert(observationHas('"item_animations":1') and observationHas('"item_animation":true'))
            assert(observationHas('"last_item_use":') and observationHas('"kind":"' .. kind .. '"'))
            assert(input() == nil and input(ButtonAction.ACTION_ITEM) == nil)
            mock.now = mock.now + 2.2; render()
            assert(observationHas('"enabled":true'))
            mock.paused = false; render()
            assert(observationHas('"enabled":true') and input() == nil)
            mock.incoming = {oldPacket}; tick(1, 0)
            assert(transportState().accepted == 1 and input() == nil)
            queue({floor_mode = true}); tick(1, 0)
            assert(input() == nil, "even an unpaused copy of the frozen frame is too old")
            tick(2, .03); queue({floor_mode = true, move = "right", shoot = "up"}); tick(2, 0)
            assert(input(ButtonAction.ACTION_RIGHT) == 1 and input(ButtonAction.ACTION_SHOOTUP) == 1)
            assert(input(ButtonAction.ACTION_ITEM) ~= 1, "item must not be used again")
        end
    end
end)
check("item unpause handles update before render without losing the fresh command", function()
    useItem("active", true); mock.paused = true; render()
    mock.now = mock.now + 2.2; mock.paused = false; tick(2, 0)
    assert(observationHas('"enabled":true'))
    queue({floor_mode = true}); tick(2, 0); render()
    assert(observationHas('"enabled":true') and input() == 1)
end)
check("manual stops and item animation expiry never resume automatically", function()
    for _, stop in ipairs({"escape", "p", "controller", "off", "death", "expiry", "room", "floor"}) do
        useItem("active", true); mock.paused = true; render()
        if stop == "escape" then mock.pauseKey = Keyboard.KEY_ESCAPE
        elseif stop == "p" then mock.pauseKey = Keyboard.KEY_P
        elseif stop == "controller" then mock.pauseAction = true
        elseif stop == "off" then mock.keys = true
        elseif stop == "death" then mock.dead = true
        elseif stop == "expiry" then mock.now = mock.now + 3.01
        elseif stop == "room" then mock.index = 11; callbacks.MC_POST_NEW_ROOM()
        elseif stop == "floor" then mock.stage = 2; callbacks.MC_POST_NEW_ROOM() end
        render(); assert(observationHas('"enabled":false'), stop)
        mock.paused = false; render(); tick(2, 0)
        assert(observationHas('"enabled":false') and input() == nil, stop)
    end
end)
check("an item packet without confirmed use cannot authorize a pause", function()
    for _, failure in ipairs({"no_callback", "no_input", "wrong_item", "wrong_player", "wrong_slot", "late"}) do
        reset(); mock.activeItem = 41; toggle()
        queue({interaction = "active", interaction_id = "unconfirmed", floor_mode = true}); tick(0)
        mock.frame = 1
        if failure ~= "no_input" then input(ButtonAction.ACTION_ITEM) end
        local target = failure == "wrong_player" and {Index = 99, InitSeed = 99} or player
        if failure == "late" then mock.frame = 4 end
        if failure ~= "no_callback" then
            callbacks.MC_USE_ITEM(nil, failure == "wrong_item" and 34 or 41, {}, target, 4,
                failure == "wrong_slot" and 1 or 0, 0)
        end
        mock.paused = true; render(); assert(observationHas('"enabled":false'), failure)
    end
end)
check("a use without an animation does not authorize an unrelated later pause", function()
    useItem("active"); tick(4, .1)
    mock.paused = true; render(); assert(observationHas('"enabled":false'))
    useItem("active"); mock.now = mock.now + .6
    mock.paused = true; render(); assert(observationHas('"enabled":false'))
end)
check("resumed item permission is bounded and cannot authorize another pause", function()
    useItem("active", true); mock.paused = true; render()
    mock.paused = false; render(); tick(2, .03)
    mock.paused = true; render(); assert(observationHas('"enabled":false'))
    useItem("active", true); mock.paused = true; render()
    mock.paused = false; render(); tick(2, 2.1)
    assert(observationHas('"enabled":false'))
end)
check("duplicate item callbacks cannot renew the animation deadline", function()
    useItem("active")
    mock.now = mock.now + .3
    callbacks.MC_USE_ITEM(nil, 41, {}, player, 32, 0, 0)
    mock.paused = true; render()
    mock.now = mock.now + 2.75; render()
    assert(observationHas('"enabled":false'))
    assert(observationHas('"last_stop_reason":"Item animation expired: F8 enables Jev"'))
end)
check("interaction cooldown suppresses pulses without dropping movement and permits later retry", function()
    reset(); toggle(); queue({interaction = "bomb", interaction_id = "first"}); tick(0); tick(1)
    queue({interaction = "pocket", interaction_id = "second", move = "right"}); tick(1); tick(2)
    assert(input(ButtonAction.ACTION_PILLCARD) == 0 and input(ButtonAction.ACTION_RIGHT) == 1)
    assert(observationHas('"interaction_status":"cooldown"'))
    tick(15); queue({interaction = "pocket", interaction_id = "second"}); tick(15); tick(16)
    assert(input(ButtonAction.ACTION_PILLCARD) == 1)
end)
check("interaction ledger is bounded at 512 IDs and never evicts old spend decisions", function()
    reset(); toggle()
    for index = 1, 513 do
        local frame = (index - 1) * 15
        tick(frame); queue({interaction = "bomb", interaction_id = "spend-" .. tostring(index)})
        tick(frame); tick(frame + 1)
        assert(input(ButtonAction.ACTION_BOMB) == (index <= 512 and 1 or 0), tostring(index))
    end
    assert(observationHas('"interaction_status":"limit"'))
    tick(7700); queue({interaction = "bomb", interaction_id = "spend-1"}); tick(7700); tick(7701)
    assert(input(ButtonAction.ACTION_BOMB) == 0)
    assert(observationHas('"interaction_status":"duplicate"'))
end)
check("interaction pulse obeys pause death session room expiry and player identity guards", function()
    for _, stop in ipairs({"pause", "death", "off", "expiry", "room", "entity"}) do
        reset(); toggle(); queue({interaction = "bomb", interaction_id = "guard"}); tick(0)
        mock.frame = 1
        if stop == "pause" then mock.paused = true
        elseif stop == "death" then mock.dead = true
        elseif stop == "off" then toggle()
        elseif stop == "expiry" then mock.now = mock.now + .56
        elseif stop == "room" then mock.index = 12; callbacks.MC_POST_NEW_ROOM() end
        local entity = stop == "entity" and {Index = 999, InitSeed = 999, ToPlayer = function() return player end} or player
        assert(input(ButtonAction.ACTION_BOMB, InputHook.GET_ACTION_VALUE, entity) == nil, stop)
    end
    for _, change in ipairs({{session = "wrong"}, {room_id = "wrong"}, {frame = 1}}) do
        reset(); toggle()
        change.interaction, change.interaction_id = "bomb", "rejected"
        queue(change); tick(0); tick(1)
        assert(input(ButtonAction.ACTION_BOMB) == nil)
    end
end)
check("superseding and neutral revoke packets cannot revive an already reserved interaction", function()
    reset(); toggle(); queue({interaction = "bomb", interaction_id = "reserved", floor_mode = true}); tick(0)
    queue({move = "none", shoot = "none", floor_mode = false, hold_frames = 1}); tick(0)
    assert(input(ButtonAction.ACTION_BOMB) == nil)
    tick(20); toggle(); queue({interaction = "bomb", interaction_id = "reserved"}); tick(20); tick(21)
    assert(input(ButtonAction.ACTION_BOMB) == 0)
    reset(); toggle(); queue({interaction = "bomb", interaction_id = "old"}); tick(0)
    tick(1); queue({move = "right"}); tick(1)
    assert(input(ButtonAction.ACTION_BOMB) == nil and input(ButtonAction.ACTION_RIGHT) == 1)
end)
check("pocket item and pickup observations hide unidentified pill effects", function()
    reset()
    mock.pill, mock.card, mock.trinket = 3, 0, 44
    mock.trinket1, mock.gigaBombs = 131, 2
    mock.bombFlags, mock.trinketModifiers = 16, {[133] = true}
    mock.pillEffects = {[3] = 7}
    mock.pillConfigs = {[7] = {Name = "#BALLS_OF_STEEL_NAME"}}
    mock.entities = {pickupEntity(1, 70, 3)}
    tick(1)
    assert(observedPickups()[1].pill_known == false)
    assert(not observationHas('"pill_effect":'))
    assert(not mock.pillEffectQueries)
    assert(observationHas('"pocket_pill":3') and observationHas('"trinket":44'))
    assert(observationHas('"trinket_1":131') and observationHas('"giga_bombs":2'))
    assert(observationHas('"bomb_flags":16') and observationHas('"unsafe_bomb_trinket":true'))
    mock.identifiedPills = {[3] = true}; tick(2)
    assert(observedPickups()[1].pill_effect == 7 and observedPickups()[1].pill_known == true)
    assert(observationHas('"pocket_name":"#BALLS_OF_STEEL_NAME"'))
    mock.card, mock.pill = 5, 0
    mock.cards = {[5] = {Name = "#THE_EMPEROR_NAME", Description = "#THE_EMPEROR_DESCRIPTION"}}
    mock.entities = {pickupEntity(1, 300, 5)}; tick(3)
    assert(observedPickups()[1].name == "#THE_EMPEROR_NAME")
    assert(observationHas('"pocket_name":"#THE_EMPEROR_NAME"'))
end)
check("inventory observation is cached refreshed on changes bounded and uses real JSON arrays", function()
    reset()
    assert(observationHas('"inventory":[]') and observationHas('"weapon_types":[1]'))
    local before = mock.inventoryQueries
    tick(1); tick(2); assert(mock.inventoryQueries == before)
    mock.inventory, mock.collectibleCount = {[1] = 2, [2] = 1}, 3
    mock.collectibles = {[1] = {Name = "#ONE", Description = "one", Type = 1},
        [2] = {Name = "#TWO", Description = "two", Type = 2, MaxCharges = 4, ChargeType = 0}}
    mock.activeItem, mock.activeCharge = 2, 3
    player.CanFly, player.ShotSpeed, player.TearRange = true, 1.2, 260
    mock.weaponTypes = {[2] = true, [9] = true}; tick(3)
    assert(mock.inventoryQueries > before)
    assert(observationHas('"weapon_types":[2,9]') and observationHas('"can_fly":true'))
    assert(observationHas('"active_charge":3') and observationHas('"active_max_charge":4'))
    local inventory = assert(mock.sent[#mock.sent]:match('"inventory":(%b[])'))
    assert(inventory:find('"count":2', 1, true) and inventory:find('"name":"#TWO"', 1, true))
    before = mock.inventoryQueries; tick(32); assert(mock.inventoryQueries == before)
    tick(33); assert(mock.inventoryQueries > before)
    mock.inventory, mock.collectibleCount = {}, 200
    for index = 1, 200 do mock.inventory[index] = 1 end
    tick(34)
    inventory = assert(mock.sent[#mock.sent]:match('"inventory":(%b[])'))
    local _, size = inventory:gsub('"count":', '')
    assert(size == 128 and observationHas('"inventory_truncated":true'))
end)
local function prepareDescent(kind)
    reset()
    mock.clear, mock.roomType = true, 5
    room.GetFrameCount = function() return mock.roomFrame or mock.frame end
    room.GetGridSize = function() return 1 end
    room.GetGridEntity = function() return {State = 1, GetType = function() return kind or 17 end} end
    room.GetGridCollision = function() return 0 end
    room.GetGridPosition = function() return {X = 320, Y = 280} end
    player.Position = {X = 320, Y = 240}
    toggle(); queue({floor_mode = true, transition = "floor", move = "down", hold_frames = 15}); tick(0); tick(1)
end
local function arriveNextFloor()
    mock.stage, mock.seed, mock.index, mock.roomType, mock.roomFrame = 2, 457, 84, 1, 1
    mock.clear = true
    callbacks.MC_POST_NEW_ROOM()
end
check("descent permission is explicit near a real clear boss exit and bounded to the next stage", function()
    for _, kind in ipairs({17, 18}) do
        prepareDescent(kind)
        assert(observationHas('"floor_descent":1') and observationHas('"floor_advance_permitted":true'))
    end
    for _, invalid in ipairs({"far", "normal", "not_clear", "rock", "teleporter", "missing"}) do
        prepareDescent()
        -- Cancel the existing token before testing a different state.
        tick(2); queue({floor_mode = true}); tick(2); tick(3)
        if invalid == "far" then player.Position.X = 500
        elseif invalid == "normal" then mock.roomType = 1
        elseif invalid == "not_clear" then mock.clear = false
        elseif invalid == "rock" then room.GetGridEntity = function() return {GetType = function() return 2 end} end
        elseif invalid == "teleporter" then room.GetGridEntity = function() return {GetType = function() return 23 end} end
        elseif invalid == "missing" then room.GetGridSize = function() return 0 end end
        queue({floor_mode = true, transition = "floor"}); tick(3); tick(4)
        assert(observationHas('"floor_advance_permitted":false'), invalid)
    end
end)
check("descent rejects malformed transition fields and requires floor authorization", function()
    for _, value in ipairs({true, 1, "none", "teleport"}) do
        reset(); toggle(); queue({floor_mode = true, transition = value}); tick(1)
        assert(transportState().last_rejection == "transition")
    end
    reset(); toggle(); queue({transition = "floor"}); tick(1)
    assert(transportState().last_rejection == "transition")
    reset(); toggle()
    mock.incoming = {(action({floor_mode = true, transition = "floor"}):gsub('"transition":"floor"', '"transition":null'))}
    tick(1); assert(transportState().last_rejection == "transition")
end)
check("normal descent retains permission across a long animation but never prior room inputs", function()
    prepareDescent()
    local old = action({floor_mode = true, transition = "floor"})
    mock.paused = true; render()
    assert(input(ButtonAction.ACTION_DOWN) == nil)
    mock.now = mock.now + 3; arriveNextFloor()
    assert(observationHas('"enabled":true') and observationHas('"floor_advance_permitted":true'))
    assert(input() == nil)
    mock.paused = false; render(); tick(2)
    mock.incoming = {old}; tick(3)
    assert(transportState().last_rejection == "identity")
    assert(input() == nil)
    queue({floor_mode = true, move = "right"}); tick(3); tick(4)
    assert(input(ButtonAction.ACTION_RIGHT) == 1)
    mock.roomFrame = 3
    queue({floor_mode = true, move = "right"}); tick(4); tick(5)
    assert(observationHas('"enabled":true') and observationHas('"floor_advance_permitted":false'))
end)
check("descent supports new-room-first and early repeated pause callback orderings", function()
    prepareDescent(); mock.now = mock.now + 3; arriveNextFloor()
    mock.paused = true; render(); mock.paused = false; render()
    tick(2); queue({floor_mode = true}); tick(2)
    mock.paused = true; render(); mock.paused = false; render()
    assert(observationHas('"enabled":true'))
    assert(input() == nil)
    mock.roomFrame = 3; tick(3); queue({floor_mode = true}); tick(3); tick(4)
    assert(observationHas('"enabled":true') and observationHas('"floor_advance_permitted":false'))
end)
check("descent reordered update waits for new-room authentication without sending mismatched state", function()
    prepareDescent()
    mock.stage, mock.seed, mock.index, mock.roomType = 2, 457, 84, 1
    local sent = #mock.sent
    tick(2)
    assert(input() == nil and #mock.sent == sent)
    arriveNextFloor()
    assert(observationHas('"enabled":true') and observationHas('"floor_advance_permitted":true'))
end)
check("descent fixed ten-second deadline cannot be extended by fresh commands or arrival", function()
    for _, timing in ipairs({"before_arrival", "after_arrival", "during_receive"}) do
        prepareDescent()
        local start = 1000
        tick(2); queue({floor_mode = true, transition = "floor"}); tick(2)
        if timing == "after_arrival" then
            mock.now = start + 5; arriveNextFloor(); mock.roomFrame = 1
        end
        mock.now = start + 9.99; tick(3, 0)
        queue({floor_mode = true, transition = "floor"})
        if timing == "during_receive" then
            local receive = udp.receive
            udp.receive = function(self, size)
                local payload, err = receive(self, size)
                if payload then mock.now = start + 10.01 end
                return payload, err
            end
            tick(4, .001)
        else mock.now = start + 10.01; tick(4, 0) end
        assert(observationHas('"enabled":false') and observationHas('"floor_advance_permitted":false'), timing)
        assert(input() == nil)
    end
end)
check("manual pause F8 death invalid destination and canceled descent revoke permission", function()
    for _, invalid in ipairs({"pause_key", "off", "death", "stage_skip", "stage_back", "boss_destination",
        "combat_destination", "same_floor_room", "cancel_pause", "ordinary_command"}) do
        prepareDescent()
        if invalid == "pause_key" then mock.pauseKey = Keyboard.KEY_ESCAPE; mock.paused = true; render()
        elseif invalid == "off" then toggle()
        elseif invalid == "death" then mock.dead = true; tick(2)
        elseif invalid == "same_floor_room" then mock.index = 12; callbacks.MC_POST_NEW_ROOM()
        elseif invalid == "cancel_pause" then mock.paused = true; render(); mock.paused = false; render()
        elseif invalid == "ordinary_command" then queue({floor_mode = true}); tick(1); tick(2)
        else
            mock.stage, mock.seed, mock.index, mock.roomType, mock.clear = 2, 457, 84, 1, true
            if invalid == "stage_skip" then mock.stage = 3
            elseif invalid == "stage_back" then mock.stage = 0
            elseif invalid == "boss_destination" then mock.roomType = 5
            elseif invalid == "combat_destination" then mock.clear = false end
            callbacks.MC_POST_NEW_ROOM()
        end
        assert(observationHas('"floor_advance_permitted":false'), invalid)
        if invalid ~= "ordinary_command" then assert(observationHas('"enabled":false'), invalid) end
    end
end)
local function setObservedGrids(grids)
    room.GetGridSize = function() return #grids end
    room.GetGridEntity = function(_, index) return grids[index + 1] end
    room.GetGridCollision = function(_, index) return grids[index + 1].collision end
    room.GetGridPosition = function(_, index) return {X = 120 + index * 40, Y = 200} end
end
local function poopGrid(variant, state, collision)
    return {State = state, collision = collision, GetType = function() return 14 end,
        GetVariant = function() return variant end}
end
check("ordinary poop reports tear eligibility from observed variant collision and damage state", function()
    reset()
    setObservedGrids({poopGrid(0, 0, 3), poopGrid(0, 250, 3), poopGrid(0, 750, 3)})
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 3)
    for index, item in ipairs(items) do
        assert(item.type == 14 and item.variant == 0 and item.kind == "grid")
        assert(item.index == index - 1 and item.x == 80 + index * 40 and item.y == 200)
        assert(item.radius == 20 and item.collision == 3 and item.tear_destructible == true)
    end
    assert(items[1].state == 0 and items[2].state == 250 and items[3].state == 750)
end)
check("destroyed other-variant and noncolliding poop remain observed without shooting permission", function()
    reset()
    setObservedGrids({poopGrid(0, 1000, 0), poopGrid(0, 1000, 3), poopGrid(1, 0, 3),
        poopGrid(2, 0, 3), poopGrid(0, 250, 0), poopGrid(0, nil, 3), poopGrid(0, 250.5, 3)})
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 7 and items[1].state == 1000 and items[1].collision == 0)
    for _, item in ipairs(items) do assert(item.tear_destructible == false) end
end)
check("missing failing or invalid grid variant APIs cannot grant tear permission", function()
    reset()
    local missing, failing = poopGrid(0, 0, 3), poopGrid(0, 0, 3)
    missing.GetVariant = nil
    failing.GetVariant = function() error("optional API unavailable") end
    setObservedGrids({missing, failing, poopGrid("0", 0, 3), poopGrid(0.5, 0, 3),
        poopGrid(0/0, 0, 3), poopGrid(-1, 0, 3)})
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 6)
    for _, item in ipairs(items) do assert(item.variant == -1 and item.tear_destructible == false) end
end)
check("only live ordinary and red fireplaces with observed positive health are shootable", function()
    reset()
    mock.entities = {}
    for variant = 0, 4 do
        mock.entities[#mock.entities + 1] = pickupEntity(40 + variant, variant, 0,
            {Type = 33, HitPoints = 5, MaxHitPoints = 6,
                ToNPC = function() return {} end, IsActiveEnemy = function() return false end})
    end
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 5)
    for index, item in ipairs(items) do
        assert(item.type == 33 and item.kind == "fire" and item.variant == index - 1)
        assert(item.id == tostring(939 + index) .. ":" .. tostring(39 + index))
        assert(item.hp == 5 and item.max_hp == 6 and item.x == 150 and item.y == 250)
        assert(item.tear_destructible == (index <= 2))
    end
    assert(#observedItems("enemies") == 0)
end)
check("dead removed healthless and malformed-health fireplaces never grant shooting permission", function()
    reset()
    mock.entities = {
        pickupEntity(1, 0, 0, {Type = 33, HitPoints = 0, MaxHitPoints = 6}),
        pickupEntity(2, 0, 0, {Type = 33}),
        pickupEntity(3, 1, 0, {Type = 33, HitPoints = 0/0, MaxHitPoints = math.huge}),
        pickupEntity(4, 1, 0, {Type = 33, HitPoints = "5", MaxHitPoints = "6"}),
        pickupEntity(5, 0, 0, {Type = 33, HitPoints = 5, dead = true}),
        pickupEntity(6, 0, 0, {Type = 33, HitPoints = 5, removed = true}),
    }
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 4 and items[1].hp == 0 and items[1].max_hp == 6)
    for _, item in ipairs(items) do assert(item.tear_destructible == false) end
    for index = 2, 4 do assert(items[index].hp == nil and items[index].max_hp == nil) end
end)
check("door metadata distinguishes curse association from actual spikes and preserves ordinary door facts", function()
    reset()
    addDoor(0, 11, 20, 280, true, false, 10)
    addDoor(1, 12, 320, 20, true, false, 1)
    addDoor(2, 13, 620, 280, false, true, 2)
    mock.doors[0].CurrentRoomType, mock.doors[1].CurrentRoomType, mock.doors[2].CurrentRoomType = 1, 10, 1
    for slot = 0, 2 do
        local variant = slot
        mock.doors[slot].GetVariant = function() return variant end
        mock.doors[slot].IsBusted = function() return variant == 1 end
    end
    tick(1)
    local items = observedItems("doors")
    assert(#items == 3)
    assert(items[1].current_type == 1 and items[1].target_type == 10 and items[1].curse_room_door == true)
    assert(items[2].current_type == 10 and items[2].target_type == 1 and items[2].curse_room_door == true)
    assert(items[3].curse_room_door == false and items[3].locked == true and items[3].open == false)
    for index, item in ipairs(items) do
        assert(item.variant == index - 1 and item.busted == (index == 2))
        assert(item.slot == index - 1 and item.target_index == 10 + index)
        assert(item.spiked == nil and item.damage == nil)
    end
end)
check("optional door APIs fail conservatively without changing existing observation schema", function()
    reset()
    addDoor(0, 11, 20, 280, true, false, 1)
    addDoor(1, 12, 320, 20, true, false, 10)
    mock.doors[0].IsBusted = function() error("optional API unavailable") end
    mock.doors[0].GetVariant = function() error("optional API unavailable") end
    mock.doors[1].IsBusted = function() return 1 end
    mock.doors[1].CurrentRoomType = "10"
    tick(1)
    local items = observedItems("doors")
    assert(items[1].curse_room_door == nil and items[2].curse_room_door == true)
    for _, item in ipairs(items) do
        assert(item.current_type == -1 and item.variant == -1 and item.busted == nil)
        assert(item.open == true and item.locked == false)
    end
end)
check("player type flight and Flat File are observations without generic invincibility inference", function()
    reset()
    assert(observationHas('"player_type":0') and observationHas('"has_flat_file":false'))
    mock.playerType, mock.trinketModifiers, player.CanFly = 21, {[151] = true}, true
    player.HasInvincibility = function() error("Generic invincibility is not a curse-door guarantee") end
    tick(1)
    assert(observationHas('"player_type":21') and observationHas('"has_flat_file":true'))
    assert(observationHas('"can_fly":true'))
    assert(not observationHas('"spike_immune":') and not observationHas('"curse_immune":'))
    player.GetPlayerType = nil
    local hasTrinket = player.HasTrinket
    player.HasTrinket = function(self, id)
        if id == 151 then error("optional trinket lookup failed") end
        return hasTrinket(self, id)
    end
    tick(2)
    assert(observationHas('"player_type":-1') and observationHas('"has_flat_file":false'))
    player.GetPlayerType = function() return "0" end
    tick(3); assert(observationHas('"player_type":-1'))
    player.GetPlayerType = function() error("optional player API unavailable") end
    tick(4); assert(observationHas('"player_type":-1'))
end)
local function roomDescriptor(index, list, changes)
    local desc = {SafeGridIndex = index, GridIndex = index, ListIndex = list,
        Clear = true, VisitedCount = 1, Data = {Type = 1}, pointer = list + 100}
    for key, value in pairs(changes or {}) do desc[key] = value end
    return desc
end

local function visitedFixture(descriptors, maps, dimension)
    local level = game:GetLevel()
    game.GetLevel = function() return level end
    _G.GetPtrHash = function(desc) return desc.pointer end
    level.GetCurrentRoomDesc = function() return maps[dimension][mock.index] end
    level.GetRoomByIdx = function(_, index, requested)
        mock.roomLookups = (mock.roomLookups or 0) + 1
        local map = maps[requested == -1 and dimension or requested] or {}
        return map[index] or {}
    end
    level.GetRooms = function() return {Size = #descriptors, Get = function(_, i)
        assert(i >= 0 and i < #descriptors)
        return descriptors[i + 1]
    end} end
    return level
end

check("visited room summaries expose only actual visits in the current dimension", function()
    reset()
    local a, unseen = roomDescriptor(10, 3), roomDescriptor(11, 4, {VisitedCount = 0})
    local mirror = roomDescriptor(10, 5, {Clear = false})
    visitedFixture({a, unseen, mirror}, {[0] = {[10] = a, [11] = unseen}, [1] = {[10] = mirror}}, 0)
    tick(1)
    local rows = observedItems("visited_rooms")
    assert(#rows == 1 and rows[1].list_index == 3 and rows[1].clear == true)
    assert(mock.sent[#mock.sent]:find('"dimension":0', 1, true))
    assert(mock.sent[#mock.sent]:find('"room_indices":[10]', 1, true))
    visitedFixture({a, unseen, mirror}, {[0] = {[10] = a, [11] = unseen}, [1] = {[10] = mirror}}, 1)
    tick(2)
    rows = observedItems("visited_rooms")
    assert(#rows == 1 and rows[1].list_index == 5 and rows[1].clear == false)
end)
check("large visited rooms use occupied engine aliases and omit the L-room gap", function()
    reset()
    local large = roomDescriptor(10, 3, {GridIndex = 9})
    visitedFixture({large}, {[0] = {[10] = large, [22] = large, [23] = large}}, 0)
    tick(1)
    local row = observedItems("visited_rooms")[1]
    assert(row.room_index == 10)
    assert(mock.sent[#mock.sent]:find('"room_indices":[10,22,23]', 1, true))
    local before = mock.roomLookups
    large.Clear, large.VisitedCount = false, 2
    tick(2)
    row = observedItems("visited_rooms")[1]
    assert(row.clear == false and row.visited_count == 2)
    assert(mock.roomLookups - before < 10, "aliases should be cached while clear state stays fresh")
    callbacks.MC_POST_NEW_ROOM(); tick(3)
    assert(mock.roomLookups - before >= 169, "new room visit must revalidate aliases")
end)
check("missing visited-room APIs omit the optional summary", function()
    reset(); tick(1)
    assert(not mock.sent[#mock.sent]:find('"visited_rooms":', 1, true))
    local a = roomDescriptor(10, 3)
    local level = visitedFixture({a}, {[0] = {[10] = a}}, 0)
    level.GetRooms = function() error("unavailable") end
    tick(2)
    assert(not mock.sent[#mock.sent]:find('"visited_rooms":', 1, true))
    assert(mock.sent[#mock.sent]:find('"protocol":1', 1, true))
end)
check("invalid visited metadata never invents clear facts", function()
    for _, change in ipairs({{VisitedCount = true}, {VisitedCount = "1"}, {Clear = 1},
            {SafeGridIndex = -1}, {ListIndex = false}, {Data = false}, {Data = {Type = "1"}}}) do
        reset()
        local a, invalid = roomDescriptor(10, 3), roomDescriptor(11, 4, change)
        visitedFixture({a, invalid}, {[0] = {[10] = a, [11] = invalid}}, 0)
        tick(1)
        local rows = observedItems("visited_rooms")
        assert(#rows == 1 and rows[1].list_index == 3)
    end
end)
check("visited summaries are bounded to the full169-cell grid", function()
    reset()
    local descriptors, map = {}, {}
    for index = 0, 168 do
        local desc = roomDescriptor(index, index)
        descriptors[#descriptors + 1], map[index] = desc, desc
    end
    visitedFixture(descriptors, {[0] = map}, 0)
    tick(1)
    assert(#observedItems("visited_rooms") == 169)
    assert(#mock.sent[#mock.sent] < 60000)
    -- Repeated descriptors cannot quietly become contradictory duplicate rows.
    descriptors[#descriptors + 1] = descriptors[1]
    tick(2)
    assert(not mock.sent[#mock.sent]:find('"visited_rooms":', 1, true))
end)
check("unverifiable descriptor pointers cannot authenticate a dimension or clear facts", function()
    for _, bad in ipairs({false, "100", math.huge}) do
        reset()
        local a = roomDescriptor(10, 3)
        visitedFixture({a}, {[0] = {[10] = a}}, 0)
        _G.GetPtrHash = function() return bad end
        tick(1)
        assert(not mock.sent[#mock.sent]:find('"dimension":', 1, true))
        assert(not mock.sent[#mock.sent]:find('"visited_rooms":', 1, true))
    end
end)
check("optional room history cannot disarm a complete dense observation", function()
    reset()
    local function dense(inventoryCount, pickupCount)
        mock.inventory, mock.collectibles, mock.entities = {}, {}, {}
        mock.collectibleCount, mock.collectibleSize = inventoryCount, inventoryCount + 1
        for i = 1, inventoryCount do
            mock.inventory[i] = 1
            mock.collectibles[i] = {Name = string.rep("N", 80), Description = string.rep("D", 160), Type = 1}
        end
        for i = 1, pickupCount do mock.entities[i] = pickupEntity(i, 100, i) end
    end
    dense(96, 32)
    toggle(); tick(1)
    local baseline = mock.sent[#mock.sent]
    assert(#baseline > 43000 and #baseline < 60000, "dense base bytes: " .. #baseline)
    local descriptors, map = {}, {}
    for index = 0, 168 do
        local desc = roomDescriptor(index, index)
        descriptors[#descriptors + 1], map[index] = desc, desc
    end
    visitedFixture(descriptors, {[0] = map}, 0)
    local before = #mock.sent
    tick(2)
    local payload = mock.sent[#mock.sent]
    assert(#mock.sent == before + 1 and not udp.closed)
    assert(#payload <= 60000 and payload:find('"enabled":true', 1, true))
    assert(not payload:find('"visited_rooms":', 1, true), "history and capability must both be omitted")
    assert(#observedPickups() == 32 and #observedItems("inventory") == 96)
    assert(payload:find('"truncated":false', 1, true))
    -- Omitting optional history must not hide a genuinely oversized base.
    dense(128, 64)
    before = #mock.sent
    tick(3)
    assert(#mock.sent == before and udp.closed)
    render()
    assert(mock.hud:find("observation too large", 1, true))
end)
local function switchFixture(items, trigger)
    room.GetGridSize = function() return #items end
    room.GetGridEntity = function(_, index)
        local spec = items[index + 1]
        local grid = {GetType = function() return spec.type or 20 end, State = spec.state}
        if spec.getVariant then grid.GetVariant = spec.getVariant
        elseif spec.variant ~= nil then grid.GetVariant = function() return spec.variant end end
        return grid
    end
    room.GetGridCollision = function(_, index)
        local collision = items[index + 1].collision
        return collision == nil and 0 or collision
    end
    room.GetGridPosition = function(_, index)
        local spec = items[index + 1]
        return {X = spec.x or 400, Y = spec.y or 160}
    end
    room.HasTriggerPressurePlates = function() return trigger end
end
check("noncolliding pressure plates export exact raw metadata and room trigger evidence", function()
    reset()
    switchFixture({{variant = 0, state = 0, x = 400, y = 160}}, true)
    tick(1)
    local items = observedItems("switches")
    assert(#items == 1)
    local item = items[1]
    assert(item.index == 0 and item.type == 20 and item.variant == 0 and item.state == 0)
    assert(item.x == 400 and item.y == 160 and item.collision == 0)
    assert(observationHas('"room_switches":1') and observationHas('"has_trigger_pressure_plates":true'))
    assert(#observedItems("hazards") == 0)
    assert(observationHas('"clear":false'), "switch metadata must not invent a clear-room state")
    switchFixture({{variant = 0, state = 3}}, true)
    tick(2)
    assert(observedItems("switches")[1].state == 3 and observationHas('"clear":false'))
end)
check("reward Greed rail and special switches retain their distinct observed variants", function()
    reset()
    local specs = {}
    for _, variant in ipairs({1, 2, 3, 9, 10}) do specs[#specs + 1] = {variant = variant, state = 0} end
    specs[#specs + 1] = {variant = 1, state = 4}
    switchFixture(specs, false)
    tick(1)
    local items = observedItems("switches")
    assert(#items == #specs and observationHas('"has_trigger_pressure_plates":false'))
    for i, item in ipairs(items) do
        assert(item.variant == specs[i].variant and item.state == specs[i].state)
        assert(item.required == nil and item.pressable == nil)
    end
end)
check("missing or invalid switch metadata is omitted rather than defaulted to unpressed", function()
    reset()
    switchFixture({{}, {variant = "0", state = "0"}, {variant = true, state = false},
        {variant = -1, state = -1}, {variant = math.huge, state = math.huge},
        {variant = .5, state = .5}, {getVariant = function() error("unavailable") end}}, true)
    tick(1)
    local items = observedItems("switches")
    assert(#items == 7)
    for _, item in ipairs(items) do assert(item.variant == nil and item.state == nil) end
    for _, bad in ipairs({1, "true"}) do
        room.HasTriggerPressurePlates = function() return bad end
        tick(mock.frame + 1)
        assert(not observationHas('"has_trigger_pressure_plates":'))
    end
    room.HasTriggerPressurePlates = function() error("unavailable") end
    tick(mock.frame + 1)
    assert(not observationHas('"has_trigger_pressure_plates":'))
end)
check("switch observation does not remove colliding plates from hazards or include unrelated grids", function()
    reset()
    switchFixture({{variant = 0, state = 0, collision = 3}, {type = 1, variant = 0, state = 0}}, true)
    tick(1)
    assert(#observedItems("switches") == 1 and #observedItems("hazards") == 1)
    assert(observedItems("hazards")[1].type == 20)
end)
check("switch array capacity and invalid geometry propagate incomplete-observation flags", function()
    reset()
    local specs = {}
    for i = 1, 32 do specs[i] = {variant = 0, state = 0} end
    switchFixture(specs, true)
    tick(1)
    assert(#observedItems("switches") == 32)
    local flags = decode(assert(mock.sent[#mock.sent]:match('"truncated_arrays":(%b{})')))
    assert(flags.switches == false and observationHas('"truncated":false'))
    specs[33] = {variant = 0, state = 0}
    tick(2)
    flags = decode(assert(mock.sent[#mock.sent]:match('"truncated_arrays":(%b{})')))
    assert(#observedItems("switches") == 32 and flags.switches == true and observationHas('"truncated":true'))
    switchFixture({{variant = 0, state = 0, x = math.huge}}, true)
    tick(3)
    assert(#observedItems("switches") == 0 and observationHas('"truncated":true'))
end)
check("optional switch bytes cannot disarm an otherwise complete dense room packet", function()
    reset()
    mock.inventory, mock.collectibles, mock.entities = {}, {}, {}
    mock.collectibleCount, mock.collectibleSize = 96, 97
    for i = 1, 96 do
        mock.inventory[i] = 1
        mock.collectibles[i] = {Name = string.rep("N", 80), Description = string.rep("D", 160), Type = 1}
    end
    local pickupCount
    for i = 1, 64 do
        mock.entities[i] = pickupEntity(i, 100, i)
        tick(i)
        local bytes = #mock.sent[#mock.sent]
        if bytes >= 57500 and bytes < 60000 then pickupCount = i; break end
    end
    assert(pickupCount and not udp.closed, "dense baseline bytes: " .. #mock.sent[#mock.sent] .. ", closed: " .. tostring(udp.closed))
    toggle()
    local specs = {}
    for i = 1, 32 do
        specs[i] = {variant = 100000, state = 100000, x = 999999, y = 999999}
    end
    switchFixture(specs, true)
    local before = #mock.sent
    tick(mock.frame + 1)
    local payload = mock.sent[#mock.sent]
    assert(#mock.sent == before + 1 and not udp.closed and #payload <= 60000)
    assert(observationHas('"enabled":true') and observationHas('"truncated":false'))
    assert(not observationHas('"switches":[') and not observationHas('"room_switches":'))
    assert(not observationHas('"has_trigger_pressure_plates":'))
    assert(#observedPickups() == pickupCount and #observedItems("inventory") == 96)
end)
check("enemy creep exports observed position size scale and timeout", function()
    reset()
    local creep = pickupEntity(51, 22, 0, {Type = 1000, Size = 24, Scale = 1.5,
        Timeout = 47, FrameCount = 12, CollisionDamage = 1,
        ToEffect = function(self) return self end})
    mock.entities = {creep}
    tick(1)
    local items = observedItems("hazards")
    assert(#items == 1 and items[1].kind == "creep" and items[1].surface == "red")
    assert(items[1].x == 150 and items[1].y == 250 and items[1].radius == 24)
    assert(items[1].scale == 1.5 and items[1].timeout == 47 and items[1].age == 12)
    assert(items[1].collision_damage == 1 and observationHas('"ground_creep":1'))
end)

check("player friendly and removed creep are not enemy hazards", function()
    reset()
    local function effect(index, variant, extra)
        extra = extra or {}
        extra.Type, extra.ToEffect = 1000, function(self) return self end
        return pickupEntity(index, variant, 0, extra)
    end
    mock.entities = {effect(1, 46), effect(2, 22, {removed = true}),
        effect(3, 23, {dead = true}), effect(4, 24, {SpawnerEntity = {
            ToPlayer = function(self) return self end}}), effect(5, 25, {SpawnerEntity = {
            ToPlayer = function() return nil end, HasEntityFlags = function() return true end}})}
    tick(1)
    assert(#observedItems("hazards") == 0)
end)

check("creep optional metadata never invents damage or invalid numbers", function()
    reset()
    mock.entities = {pickupEntity(1, 26, 0, {Type = 1000, Scale = 0/0, Timeout = -1,
        CollisionDamage = math.huge, ToEffect = function(self) return self end})}
    tick(1)
    local item = observedItems("hazards")[1]
    assert(item.kind == "creep" and item.timeout == -1)
    assert(item.scale == nil and item.collision_damage == nil)
end)

print(tostring(count) .. " mod tests passed; engine and bundled JSON/UDP integration remain untested.")
