from gym.envs.registration import register

register(
    id='FloorPositioning-v0',
    entry_point='environments.floor:FloorEnv',
)

register(
    id='DeathValley-v0',
    entry_point='environments.death_valley:DeathValleyEnv',
)
