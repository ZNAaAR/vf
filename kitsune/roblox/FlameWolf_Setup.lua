-- FlameWolf_Setup.lua
-- Run once in Roblox Studio after importing FlameWolf.fbx:
--   1. Click the imported FlameWolf model in the Explorer.
--   2. Paste this whole file into the Command Bar (View > Command Bar) and press Enter.
-- It anchors the parts, makes the flames and tails non-solid, and adds glow lights and embers.
-- Run it only once (running it again adds a second set of lights and embers).

local Selection = game:GetService("Selection")
local model = Selection:Get()[1]
assert(model and model:IsA("Model"), "Select the imported FlameWolf model in the Explorer first")

local function find(name)
	local part = model:FindFirstChild(name, true)
	if not part then
		warn("FlameWolf setup: part not found: " .. name)
	end
	return part
end

-- solid body, see-through-to-physics flames and tails
for _, part in ipairs(model:GetDescendants()) do
	if part:IsA("BasePart") then
		part.Anchored = true
		if part.Name:match("^Flames_") or part.Name:match("^Tails_") then
			part.CanCollide = false
			part.CanTouch = false
			part.CanQuery = false
		end
	end
end

-- glow: Roblox has no emissive texture slot, so the flames get real lights
local function addLight(part, color, range, brightness)
	local light = Instance.new("PointLight")
	light.Color = color
	light.Range = range
	light.Brightness = brightness
	light.Shadows = false
	light.Parent = part
end

local fire = find("Flames_Fire")
local mane = find("Flames_Mane")
if fire then
	addLight(fire, Color3.fromRGB(255, 120, 40), 18, 2)
	local embers = Instance.new("ParticleEmitter")
	embers.Name = "Embers"
	embers.Color = ColorSequence.new(Color3.fromRGB(255, 190, 80), Color3.fromRGB(255, 70, 20))
	embers.LightEmission = 1
	embers.Size = NumberSequence.new({NumberSequenceKeypoint.new(0, 0.25), NumberSequenceKeypoint.new(1, 0)})
	embers.Transparency = NumberSequence.new(0, 0.3)
	embers.Lifetime = NumberRange.new(1, 2)
	embers.Rate = 18
	embers.Speed = NumberRange.new(2, 5)
	embers.SpreadAngle = Vector2.new(25, 25)
	embers.Acceleration = Vector3.new(0, 4, 0)
	embers.EmissionDirection = Enum.NormalId.Top
	embers.Parent = fire
end
if mane then
	addLight(mane, Color3.fromRGB(255, 170, 60), 14, 1.5)
end
local eyes = find("Wolf_Eyes")
if eyes then
	addLight(eyes, Color3.fromRGB(255, 200, 80), 5, 2)
end

print("FlameWolf setup done")
