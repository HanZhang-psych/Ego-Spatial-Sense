# Ego Spatial Sense

For each of 360 ego-centered directions:

1. LiDAR keeps the nearest object distance in that direction.

2. The previous and current distances are passed through a learned
   distance-sensitivity / attention-like transform.

3. The model computes the change between transformed current and
   transformed previous distance.

4. That change signal is passed through a small learned projection.

5. The projected change is multiplied by the current transformed
   distance signal, so looming matters most when the object is currently
   close/relevant.

Result: one ego spatial sense value for that direction.


```{python}
prev = attention_like_transform(previous_distance)
curr = attention_like_transform(current_distance)

looming = curr - prev
looming = learned_projection(looming)

ego_spatial_sense = looming * curr
```

## Difference From The Visual Search Model

The visual search model and the ego spatial sense model have a similar
architectural shape:

```text
raw spatial input
-> ego-centered spatial evidence
-> attention/window-like gating
-> priority or sense field
-> choice or action
```

The difference is what counts as evidence and what the readout predicts.

In visual search, the input is a mostly static display. The model builds
priority over candidate items or locations from color contrast, shape
match, task goals, and selection history. That evidence is gated by an
attention window centered on fixation, then a softmax predicts which item
the eyes will choose:

```text
feature contrast + goal + history
over objects/locations
-> softmax choice
```

In ego spatial sense, the input is a dynamic LiDAR stream. The model
builds a directional field from current proximity and change over time:
whether the nearest object in each direction is currently close and
whether it is looming. A neural action head then maps the 360-direction
sense vector to a continuous movement action:

```text
proximity + looming
over movement directions
-> continuous action
```

So visual search asks: which object or location should I look at?

Ego spatial sense asks: which movement direction is spatially safe or
urgent right now?

The shared idea is that both systems construct an ego-centered spatial
field before acting. Visual search constructs a priority field over
items; ego spatial sense constructs a pressure/sense field over movement
directions.
