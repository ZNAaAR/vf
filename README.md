# Desert Shrinker (Roblox Studio plugin)

Makes your desert map smaller and more compact **without removing the boss**.
The boss (and, if you want, every other NPC) keeps its normal size and is moved
to the matching spot in the smaller desert, standing on the ground.

## Install

1. Download `DesertShrinker.lua` from this repo.
2. In Roblox Studio, open the **Plugins** tab and click **Plugins Folder**.
3. Put `DesertShrinker.lua` in that folder.
4. Restart Studio. A **Desert Shrinker** button shows up in the Plugins tab.

## Use

1. **Save a backup of your place first** (File → Save to File As...).
2. Click **Desert Shrinker** to open the panel.
3. In the Explorer, click the Model or Folder that holds your desert, then click
   **Use selected as Desert**.
4. The boss is found automatically: any model with a Humanoid whose name contains
   "boss". If it says "(none)", click your boss in the Explorer and click
   **Use selected as Boss**.
5. Type the new size, for example `50` for half the size.
6. Pick your options:
   - **Mode: Shrink everything** makes the whole desert smaller (ground, rocks,
     cacti, buildings).
   - **Mode: Pull together** keeps rocks and props at their size but moves them
     closer together. Big flat ground parts get narrower.
   - **Also shrink Terrain** turns on if your sand and dunes are made with the
     Terrain editor and not with Parts.
   - **Other NPCs keep normal size** stops regular enemies from shrinking too.
7. Click **Shrink Desert!** Press **Ctrl+Z** if you don't like the result.

## Good to know

- Terrain is only shrunk inside the box around the desert's parts. If your desert
  is *only* terrain, add a big Part that covers the whole desert, underground
  included, and put it in the desert folder. Delete that Part afterwards.
- When the terrain shrinks, the space where the old, bigger desert was becomes
  empty.
- If a script spawns the boss (or players) at fixed coordinates, update those
  coordinates to match the new, smaller desert.
