# Camera settings for sharp fast balls (plain language)

Joe asked for this in layman's terms, and noted the table lighting cannot
change. So this only uses dials on the camera.

## The problem, in one line

When you hit a ball hard, the camera's shutter is open long enough that the
ball MOVES while the picture is being taken. It comes out as a faint smear
instead of a solid ball, and the software cannot find a smear.

Measured on your own footage (the 4 ball on 20260820-005048 @233):

| | |
|---|---|
| how fast the ball was going | about 8 feet per second (a medium shot, not a break) |
| how far it smeared | about 17 pixels, on a ball only 28 pixels wide |
| what that implies your shutter is set to | about **1/75 of a second** |

A ball smeared by more than half its own width stops looking like a ball.
That is why the software lost it for half a second.

## The one number to change

**Shutter speed. Go from 1/75 to 1/250.**

| shutter | smear | verdict |
|---|---|---|
| 1/75 (now) | 17 px | ball is a faint streak — software fails |
| **1/250 (target)** | **5 px** | ball stays a solid dot — software works |
| 1/500 (ideal) | 2.6 px | sharper still, but costs more light |

1/250 is the goal. 1/500 is better but harder to pay for. Do not bother
going past 1/500 — nothing improves and the picture just gets darker.

## What it costs, and how to pay for it

A faster shutter means the shutter is open for LESS time, so LESS light
gets in, so the picture gets darker. Going 1/75 -> 1/250 makes the picture
about **1.7 "stops" darker**. A "stop" just means half as much light.

You have two ways to make up for it, since the room lights are fixed:

**1. Open the lens wider (do this first — it is free).**
The lens has an f-number: f/3.5, f/5.6, and so on. CONFUSINGLY, a SMALLER
f-number means a WIDER opening and MORE light. Going from f/5.6 to f/3.5
roughly doubles the light — that is most of what you need, at no cost in
picture quality. Shooting straight down at a flat table, there is no
downside to opening the lens all the way. **Set it to the smallest
f-number your lens allows.**

**2. Raise the ISO (use this for whatever is left).**
ISO is how sensitive the sensor is. Doubling it (400 -> 800 -> 1600)
doubles the brightness, but adds grain. Grain looks worse to your eye than
it does to the software — and our recording is shrunk down before analysis,
which averages a lot of grain away. **Grain costs us very little. Blur
costs us everything.** If you must choose, choose grain.

A likely working combination: lens wide open, ISO around 1600, shutter
1/250.

## What NOT to change

- **Frame rate.** The T3i cannot send 60fps over HDMI — that is a hardware
  limit, not a setting. Leave it at 30.
- **Anything else.** Picture style, sharpness, and so on make no difference
  here.

## How we will check it worked

Shoot two or three firm shots at the new settings. Send nothing — just tell
me the session is there. I will measure the smear and the detection rate
the same way I measured the numbers above, and tell you plainly whether it
is better, and whether the grain cost us anything.

## Important

None of this fixes the 35 sessions already recorded. Those need the
software recovery, which is a separate piece of work. Camera settings make
every FUTURE session easier; they cannot go back in time.
