# v0 — eval summary

Photos: 47 handstands + 7 controls · runs per photo: 5 · total analyses: 270

| metric | value |
|---|---|
| run-level agreement with Shirley (handstands) | 140/193 = **73%** |
| photo-level (majority of runs) agreement | 28/47 = **60%** |
| consistency: same label on every run | 39/47 = **83%** |
| unclear / error runs (handstands) | 42/235 |
| mean cost per analysis | $0.0046 (total $1.24) |
| mean latency per analysis | 2.5 s |
| models that answered | {'gpt-4o': 264, 'gpt-4-turbo': 6} |

## Confusion (photo-level, Shirley → model majority)

| Shirley \ model | good form | banana back | other |
|---|---|---|---|
| good form | 13 | 6 | 4 |
| banana back | 5 | 15 | 4 |

## Agreement by axis (photo-level majority)

**view**  
- angled: 1/1 (100%)
- back: 1/3 (33%)
- front: 4/9 (44%)
- side: 22/34 (65%)

**support**  
- freestanding: 26/41 (63%)
- wall: 2/6 (33%)

**quality**  
- blurry: 1/1 (100%)
- dark: 3/6 (50%)
- good: 23/37 (62%)
- low_res: 1/3 (33%)

**mirror**  
- no: 27/46 (59%)
- yes: 1/1 (100%)

## Controls (not a handstand) — what did the model say?

| photo | labels over runs |
|---|---|
| 041_front_freestanding_control_standing-yoga-leg-ho.jpg | {'unclear': 5} |
| 042_side_freestanding_control_surfer.jpg | {'unclear': 5} |
| 043_side_freestanding_control_crouched-arm-balance.jpg | {'unclear': 5} |
| 044_side_freestanding_control_bronze-statue-of-han.jpg | {'banana back': 5} |
| 045_angled_freestanding_control_astronaut-floating-u.jpg | {'unclear': 5} |
| 046_side_freestanding_control_plank-on-yoga-mat.jpg | {'unclear': 5} |
| 053_side_freestanding_synthetic-control_mirror-pike.jpg | {'unclear': 5} |

## Photos to read for error analysis (majority disagrees with Shirley, or runs disagree with each other)

| photo | Shirley score | Shirley label | model labels over runs |
|---|---|---|---|
| 004_side_freestanding_gym.jpg | 7 | banana back | {'unclear': 5} |
| 006_side_freestanding_rock-summit.jpg | 8 | banana back | {'good form': 5} |
| 010_side_freestanding_garden-autumn.jpg | 3 | good form | {'good form': 3, 'banana back': 2} |
| 011_side_freestanding_park.jpg | 5 | good form | {'banana back': 3, 'good form': 2} |
| 013_side_freestanding_parkour.jpg | 3 | good form | {'banana back': 3, 'good form': 2} |
| 014_side_freestanding_kettlebell-base.jpg | 3 | good form | {'banana back': 4, 'good form': 1} |
| 020_side_freestanding_football-freestyler.jpg | 4 | good form | {'unclear': 5} |
| 022_side_freestanding_railway.jpg | 8 | banana back | {'unclear': 2, 'banana back': 3} |
| 023_side_freestanding_child-gymnast.jpg | 6 | banana back | {'good form': 5} |
| 026_side_freestanding_night-b-w.jpg | 5 | good form | {'banana back': 5} |
| 027_side_freestanding_lake-at-dusk.jpg | 6 | banana back | {'good form': 5} |
| 028_side_freestanding_red-rock-canyon.jpg | 3 | good form | {'banana back': 5} |
| 029_front_wall_brick-wall.jpg | 7 | banana back | {'unclear': 5} |
| 030_front_wall_against-wooden-door.jpg | 5 | good form | {'unclear': 5} |
| 032_back_wall_pool-wall.jpg | 6 | banana back | {'unclear': 5} |
| 035_front_freestanding_gymnast-front-close-up.jpg | 6 | banana back | {'banana back': 1, 'unclear': 4} |
| 036_back_freestanding_gymnast-seen-from-behind.jpg | 6 | banana back | {'good form': 5} |
| 038_front_freestanding_beach.jpg | 4 | good form | {'unclear': 4, 'banana back': 1} |
| 039_front_freestanding_yoga-mat.jpg | 3 | good form | {'unclear': 5} |
| 040_front_freestanding_studio-front.jpg | 3 | good form | {'good form': 3, 'unclear': 2} |
| 049_side_freestanding_locker-room_derived-lowres.jpg | 6 | banana back | {'good form': 5} |
| 052_side_wall_synthetic-dark-garage.jpg | 4 | good form | {'banana back': 5} |
