import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium", app_title="Holstein")


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Holstein Checkerboards

        The [Holstein rule](https://conwaylife.com/wiki/OCA:Holstein) is a totallistic cellular automaton with births for 3,5,6,7 or 8 neighbours and survival for 4,6,7 or 8 neighbours. [Paul Rendell](http://rendell-attic.org/) has experimented with using [checkerboards](http://rendell-attic.org/CA/holstein/checkerboard.htm) with edge defects as initial states. Perpetrated in [Python](https://www.python.org/)/[Marimo](https://marimo.io/) by abusing [`scipy.signal.convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html), [`numpy.isin`](https://numpy.org/doc/stable/reference/generated/numpy.isin.html), [`numpy.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) and [Pillow](https://pillow.readthedocs.io/en/stable/). See also: [Designing Beauty: The Art of Cellular Automata](https://link.springer.com/book/10.1007/978-3-319-27270-2).

        If you have [`ffmpeg`](https://www.ffmpeg.org/) you can churn out a video with: `python holstein.py --fname video.mp4`
        where options include:

        `--width`: image width, default 512  
        `--period`: grid width, default 64  
        `--defects`: number of edge defects, default 7  
        `--nframes`: number of frames, default 2028
        `--skip`, stepsize between frames, default 2  
        `--framerate`: default 30
        """
    )
    return


@app.cell
def __():
    from itertools import islice
    import random
    try:
        import subprocess as sp
    except:
        pass

    import marimo as mo
    import numpy as np
    from PIL import Image
    from scipy.signal import convolve2d
    return Image, convolve2d, islice, mo, np, random, sp


@app.cell
def __(np):
    def checkerboard(dim, period):
        rnge = np.arange(dim)
        x, y = np.meshgrid(rnge, rnge)
        return (np.logical_xor((x // period) % 2, (y // period) % 2) & 1).astype(np.uint8)
    return (checkerboard,)


@app.cell
def __(checkerboard, np, random):
    def perturb_checkerboard(dim, period, n=1):
        grid = checkerboard(dim, period)
        squares = dim // period
        for _ in range(n):
            x = period * random.randint(0, squares-1)
            y = period * random.randint(0, squares-1)
            if random.random() > 0.5:
                x += period // 2
            else:
                y += period // 2
            grid[x, y] = np.logical_not(grid[x, y]) & 1
        return grid
    return (perturb_checkerboard,)


@app.cell
def __(convolve2d, np):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    def apply_ca_rule(cells, born, survive):
        neighbours = convolve2d(cells, kernel, mode='same', boundary='wrap')
        next_cells = np.zeros(cells.shape, dtype=np.uint8)
        x, y = np.where(cells == 1)
        next_cells[x, y] = (np.isin(neighbours[x, y], survive) & 1).astype(np.uint8)
        x, y = np.where(cells == 0)
        next_cells[x, y] = (np.isin(neighbours[x, y], born) & 1).astype(np.uint8)
        return next_cells
    return apply_ca_rule, kernel


@app.cell
def __(apply_ca_rule, np):
    class TotalisticCA(object):

        def __init__(self, born, survive):
            self.born = np.array(born)
            self.survive = np.array(survive)

        def run(self, state):
            yield 255 * state
            new_state = apply_ca_rule(state, self.born, self.survive)
            while True:
                yield 255 * new_state
                new_state = apply_ca_rule(new_state, self.born, self.survive)

        def history(self, state):
            states = self.run(state)
            im = states.__next__()
            yield im
            while True:
                im = np.bitwise_or(states.__next__(), np.right_shift(im, 1))
                yield im
    return (TotalisticCA,)


@app.cell
def __(TotalisticCA):
    holstein = TotalisticCA([3,5,6,7,8], [4,6,7,8])
    return (holstein,)


@app.cell
def __(Image, np):
    def green_image(im):
        green = np.zeros(im.shape+(3,), dtype=np.uint8)
        green[:, :, 1] = im
        return Image.fromarray(green)
    return (green_image,)


@app.cell
def __(green_image, holstein, islice, perturb_checkerboard, sp):
    def render_video(width, period, defects, fname, nframes=2048, skip=2, framerate=30):
        ffmpeg_cmd = [
            'ffmpeg', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(framerate),
            '-i', '-', '-vcodec', 'libx265', '-qscale', '0', fname
        ]
        frames = map(
            green_image,
            islice(
                holstein.history(perturb_checkerboard(width, period, defects)),
                0, nframes, skip
            )
        )
        pipe = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE)
        for im in frames:
            im.save(pipe.stdin, 'PNG')
        pipe.stdin.close()
        pipe.wait()
    return (render_video,)


@app.cell
def __(mo):
    grid_dropdown = mo.ui.dropdown(
        options=[str(2**i) for i in range(1, 8)], value='8', label="grid squares"
    )

    defect_slider = mo.ui.slider(start=0, stop=64, step=1, value=5, label='defects')
    return defect_slider, grid_dropdown


@app.cell
def __(
    defect_slider,
    green_image,
    grid_dropdown,
    holstein,
    perturb_checkerboard,
):
    img_seq = holstein.history(perturb_checkerboard(512, 512//int(grid_dropdown.value), defect_slider.value))
    def next_img(_):
        return green_image(img_seq.__next__())
    return img_seq, next_img


@app.cell
def __(mo, next_img):
    next_button = mo.ui.button(value=next_img(None), on_click=next_img, label='step')
    return (next_button,)


@app.cell
def __(defect_slider, grid_dropdown, mo, next_button, render_video):
    default_args = {
        'width': 512, 'period': 64, 'defects': 7, 'nframes': 2048, 'skip': 2, 'framerate': 30
    }

    if __name__ == '__main__':
        args = mo.cli_args()
        if not args:
            ui = mo.vstack([grid_dropdown, defect_slider, next_button, next_button.value])
        else:
            ui = None
            fname = args['fname']
            if fname:
                vid_args = {'fname': fname}
                for k, v in default_args.items():
                    if args[k]:
                        vid_args[k] = int(args[k])
                    else:
                        vid_args[k] = v
                render_video(**vid_args)        
    return args, default_args, fname, k, ui, v, vid_args


@app.cell
def __(ui):
    ui
    return


if __name__ == "__main__":
    app.run()
