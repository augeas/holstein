## Holstein Checkerboards

The [Holstein rule](https://conwaylife.com/wiki/OCA:Holstein) is totallistic cellular automata with births for 3,5,6,7 or 8 neighbours and survival for 4,6,7 or 8 neighbours. [Paul Rendell](http://rendell-attic.org/) has experimented with using [checkerboards](http://rendell-attic.org/CA/holstein/checkerboard.htm) with edge defects as initial states. Perpetrated in [Python](https://www.python.org/)/[Marimo](https://marimo.io/) by abusing [`scipy.signal.convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html), [`numpy.isin`](https://numpy.org/doc/stable/reference/generated/numpy.isin.html), [`numpy.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) and [Pillow](https://pillow.readthedocs.io/en/stable/). See also: [Designing Beauty: The Art of Cellular Automata](https://link.springer.com/book/10.1007/978-3-319-27270-2).

If you have [`ffmpeg`](https://www.ffmpeg.org/) you can churn out a video with: `python holstein.py --fname video.mp4`
where options include:

`--width`: image width, default 512  
`--period`: grid width, default 64  
`--defects`: number of edge defects, default 7  
`--nframes`: number of frames, default 2028  
`--skip`, stepsize between frames, default 2  
`--framerate`: default 30  

Play with the Marimo notebook via web-assembly [here](https://marimo.app/#code/JYWwDg9gTgLgBCAhlUEBQaD6mDmBTAOzykRjwBNMB3YGACzgF44AiABgDoBODgJgGYWaRGDBMEyVBwCCogBQ1y9RixAVgAVxAsANHBFhMMWgBs8KgBIQTAZzLACLAJQY0AAQMcAxnhMm05HgAZnDYciAQTgBcaHBxCBAcIORysfHpUCxZaelxAMR5cFa29gRwAMJ0eF4A1sQARhDI5DYYufEAKlVwANrFdngOcFAaZgC6cnQwMGA2UQD0814QBFSIAJ4mwEF43hAg8zQ1wPMA8uXSUf2lTnDANnAwEDCIfvfGXnA+fqPI+hpPJAvOA0ehwerAWB0B5BaBwfg6ACsOgAbDoAOxwOEADjgRGAODojQ0UAeiAI5DgNhJADdgDTXnBYVA4AAWVEYrEs3H4wnE0kcXoABUQozgACVCIE-BMpjMFvMoFLfCYALSkD4caA4ea3OiIB54AAeYGIoEIZEpoIYGhsDhwvS8VVqDSaUBasumYAVSopKvV02AXi1UB1F3mdGsAwcS2ddSgjWaHCmIFu1rgFHwcECOy8MDJDwctGAjLspDwNkFQuIppgJEtdzKPSF63oK09MzmiyoPY4YFbkYIIZ1TnmPQAspIIh3ZgqkCgIhxgBBdeD1vp6rb7b0AAY2LzAfscO04AivPYEGnWGl4XjkHczrvzcgQLyV-eH9bD5+v+Yf-uKsExCED48z4EQ9YUH+B5HieZ4mBeV4mDed7JjAIAmE4eg9DuBBaEe9wOA+kxek+eHgF+2o-l4f4vPUZiATsvqgeBxDluQ8zkQRdpDimmHYbh+FflQVRKsRcqzosXGUaG1G0Yg9F4IxwEECxhBsZanFCRwInAWhGG3OSlLNsAfgQFQj4KmApkmOZHBKogShVC+b5LiuhDyYpuqCgAyngeD6LYEBRL0AAiFYEgQ24AEJ4KKMDrCFXQBdIsBYiE5Qqr8LLSAC+ykIglmLFsBA1MeYAoAQ+BQHsByNBANTzAAjJwLVsOi8xcOi2KqvwvVNVwqq8Oiw1sENTgcG07RwAAkiE6wQBocD6jeu5BEE4B4Dg4mkQqPZUBw62bTg363AtS1eOSXx0CSZSLfAiBwHSgQQCCtB0CFO79m2ZSRiUgxDv2cCqqqQRnmoT3AC9SRgKyO45LkulKliYDGCshaqSYGiBDECPpDuIOKPQO4haAiBZkTdB6DmoomPAiJNbwcR4-EBOqqaKAQPeIU4CgVpQ-Q1PBLT8AoqyzPTXAbM5tU+Yk3iWj1MQ6UZuQWYy3mNhC0EItwJicAs3EbMEEEJBqDY8vkUrLIQCEpuIOb2u67wbC8NihtSyDNjHGAO56AMszAAAXgFSswFQ-llPbjvZsLozwEzBuS2z0caXg8s0-H8JsB7WRCOkLgZHgMC3a4HiiN4KoBMEoSYHI0QI6b+x3GQsAQFGdzgNA8D3FsPgI6AkBpSQFL7AjdaJR7g-d1SGj1BVr4VgWVJgAjxo+KjMSS2ABqtAPXdpfOqD6A8ET70P8DSSfeKr+kTcgHAQozQAMp3F+zUg+CN1Azf-l+cGMmnmlZYl5ry3nIAjJUJcoBlBmp-PAegQFIRQuQPQvcgwIISHoAgYA9AjxfCAf2q8MDlzAJXPw1cQhhBwQ3dIOZrrVHjImd0chyCgD0BzZc5BaHTRgVmZgOCODIHJPgVhoBC7TSNHodcAiyHmzoLzKGcg+GYJURI9oUDbpwDkII2yOAgyvEwEaaAcg5BGjgIsOAnCua3AAKRwF4HoOQ65LHWO4XAexvBbgADI4BNQmgaBKpodFkI0A4GA2J1HDGLlouQTpGGumaDoFwJDPDfH8PQsI8SXQJjdKgm+eCjL7B4XEehHNoH1EwNkpheSxGEKsWaLm2DGD+K3tNRRlJmDVMSSwth9S3FRPSDYAAjhoZAFZxB9IsfMBpnMIGS2ZKERswwRF4B0SUyWcBzHMDcXAAAVCs0eIB7JGXCXINg-tRnjJsKqfxHt0gyNmVw-ZhyCEnIpGci5VIrlKhuXczZ8RtivP2O8gh9c4AAD44CcERG0gF6RzEAGodmNMpJY3g9z4i+BsHgOF8K4jrmRU8rm0yHGYriB0noUi4DrDGOIXREB9GXRMJgAgzw5CUupbSnxfiPaaJgXADpkCYkCrkOUkklTum5KSSk9waSq6ZLrkgsBd5sFgA2fGIgJh6VkJWBWUxCJ4RYWzEE8wgiwkEAiVEzVvgehNT0E1OlzAc4I3oQYEw6wqmIEwCMMwcSVRa3BNAAg-taT0jwBs9IvIiSLVJOIZVyFwH+r8IGm1Jg9ARECIwAA5DYB2eBs16GJBSZA6wc1UBIGAbNgz4hECNDAKpAadUcBDj-GwybbDHn1KaamprGDmvCZEj2XLm1IzWekh4jBmD-OmnWhtE6qXSKddowRhECA6MGHyWNNhF00rGKGqAdIbw8v8UIuw6xgkDstUOyWI7ZE6VEuOptU7oU1riHOxtKbd20vECEpcPEN0EhjSSHdXL91BpgSegJ57L2hMHW+6J0CygfoncKpD+hRAeq9T60YmC01l3lRQxVch3WesujhswaqNleBMLvOAHRnivHeEGC4cgID1AAFaywbh7RVRYG11xxSYIIRbg0HqPRGvFkshNBA4I0AV97kAkHWHIeTBAENDN8LJ6kh7w3NqUxsOQOmJMpMlvQkY66ZP+xeGQSN011jAF8JSXgiJEQvLLGQcleI8BUEwB5gKzBSPYd9Ws-z-stNybE1SCLxnw0afiCJUyAUOgjFxV5h5jmTDOdcy8ogvn-PpdrT5vzNmAsYbAFh8jIWN35dK+F4TkWYH1e02G49U12j0LoO8aAKmrNUlK3Z9o-nJ3RYaxZozA2vOgHEMNjg2AP113iwSzLlJQBecS2YejqWpPwum-eiE4d7h4EwCY2b83jQCfrmq+yQGG02C60EGAch2F+KcEt3IDmnOdzQ7EhjLwth2BY9IZJBGK7pMobXOQf2mOA68BcDZf1oxlGYNDgHHxWM9ARMiNE6IdDYnAz0dkOO8djCifysokwoylBB6ksHCqa5hDgeTTBNC8X0N5pHTAZNRHiJ2xSpUhBm2togO20AXaRB4ERXIBExqlAXrNXB69CGOeEB6FEPQ6u-HLrW5LcnH9meHR-vOZTHKBfqZ+6KlXBAufwJp3KunRGGd1ytzb5nehEfU7uDYPumDxUJiqXGHp+SbDqrZzXX0gQoCYGengCACgBZU2JfkjW+Y9Bg3zdg1ONhGAu1ZNif2Psc9p7NmnRg-A2CDaZBtU0OAqnJHED0Lz2ajo18LXAbNoM2-Zu57eaypou+qhpMsQIXgu9gCqgPqAbe7BQDkKnSCWEm+qmAAPgfQ+ubVC71seoRpeAokRAP4Z+5XgFr0Nmtgbf09qA9mMD2WfxBIDAKkfFLue86Cm97jBz-8XxA9wDZM3WUAKmfukqge0qLClMHCqK2sssNgb27+P+8QXyJsJega3sh4XmCGCGfeZWIeHAQoEApo66LeW0deweMAbCBAjAeBT8QoAAoghostNkMFnnzrkGLnmjeGKoeLsHYJQWfkKAAHIADi1aHsOBx4FBDg3gtkOK9c4hPBOkiAtA8hRc6GyiyoUeMeEAdupC5CGSTu4QkQeKHSmA5AP8YAL4VAyOCQHAYSHA5hhBVh66HshBaMBA2ePQM+cgvAeyeywAtwTByyI8oi9qcAkS4GDIWM5g2a2IbetGSsJgKgHS3yYyvy+c8Qpm8QKefmWwkeD+iQ9hn+keE2yAMAjAXydghBjAYs1meAYALSegURGg5gyIcACRvgOaKeNgYhahWiORxRxAegphjhlh5kBAoOZC4OxGrqQEeYuRUMQxCMr+tuyxfMZhFhzhCBv+VOAM2xcQIBAeCS4BqCaAGyoAteOKwy4gf+0hXWVRQBYqxAFSRxOSzCKQDMjicAnxiwZyIxmx4xHAzREaMB8xgxNUwJb2sxIQH6FxcgmAleeuKxzOz2IAlxeAwyc2rKF2YQUJfRAqFxfmGJ2COJFxkx+hEOYQEQJJ9aNuOAGyH6m40wKwBRdhwAcmAITw66wJ-apJaJcggheqxqKwVSfcNQvJtJFxegHRSRuaZAVaZOIqFOjJnJKwuhhGBhVCdcAxeRQxgq6xoxWxWC3mtJTJXJeCmh0eixxh0JIsmAyAOAI2AA3h7NmpTNmiFJ8Wfm4h6XALUe3t0b6bju3igfmj0SFLnvnu3ugVWhGWfvPuWL6eXgjAAL7tZ3BangzHaYBMDMDZrYBIAODYAekewOkjaLg0bAD2mhjtoIZApsoPQ1lsHpBhKsk0hli1ByA9D-FOHjGgl3a6lQA0kNpmlqkmkjmqlDjAmk4ezYppb4qtnMCClEBeZX5lZlk9DN5ZnZq374pAprnNnTTPTVmOniBOlbn5q+lrlpmIFxCLI1BNHLKZx0wnmVi0B4AgC1mHkApAobk1BjDfk-7Hl-na6WokY1k9D-nvaSxzmAX4rAUQX-niA0iFZxAR7EBWkvRyB+EIWOm3DTQW5lBllOzxyvlp5Zl6APlwBhJNFNFQyvnknTGGFhIbJhKEWuBArzb5rYC5msAFnKHW6YAsB4qeDjZOBAA?mode=read&show-code=false)


