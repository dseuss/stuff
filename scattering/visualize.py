#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from vispy import app, visuals
from vispy.visuals.transforms import STTransform
from time import time
from bubblebox import BubbleBox


class BallBoxVisualizer(app.Canvas):
    def __init__(self, bubble_box, **kwargs):
        app.Canvas.__init__(self, keys='interactive', **kwargs)
        self._bubble_box = bubble_box

        new_scale = min(size_w / size_bb - 0.5
                        for size_w, size_bb in zip(self.size, self._bubble_box.boxsize))
        new_center = np.array(self._bubble_box.boxsize)[::-1] / 2
        transform = STTransform(scale=(new_scale, new_scale),
                                translate=new_center)

        self.markers = visuals.MarkersVisual()
        self.update_positions(0.0)
        self.markers.transform = transform

        w, h = self._bubble_box.boxsize
        bbox = np.array([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], np.float32)
        self.box = visuals.LineVisual(pos=bbox, width=1, color='black',
                                      method='gl')
        self.box.transform = transform

        self.visuals = [self.markers, self.box]
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_draw(self, event):
        self.context.clear(color='white')
        for vis in self.visuals:
            vis.draw()

    def on_timer(self, event):
        elapsed = time() - self._basetime
        self.update_positions(elapsed)
        self.update()

    def on_mouse_wheel(self, event):
        """Use the mouse wheel to zoom."""
        for vis in self.visuals:
            vis.transform.zoom((1.25**event.delta[1],)*2, center=event.pos)
        self.update()

    def on_mouse_move(self, event):
        if event.is_dragging:
            dxy = event.pos - event.last_event.pos
            button = event.press_event.button

            if button == 1:
                for vis in self.visuals:
                    vis.transform.move(dxy)

        self.update()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for vis in self.visuals:
            vis.transforms.configure(viewport=vp, canvas=self)

    def on_key_press(self, event):
        pass

    def update_positions(self, t):
        t = t / 10
        positions = self._bubble_box.positions(t)
        self._bubble_box.propagate(t)
        sizes = 2 * self._bubble_box.ballsize
        self.markers.set_data(positions, size=sizes, scaling=True,
                              face_color='red', edge_width=0)


if __name__ == '__main__':
    app.use_app(backend_name='pyglet')
    bubble_box = BubbleBox((10, 10), number=60, temperature=10)
    #  bubble_box._pos = np.array([(1, 1)])
    #  bubble_box._vel *= 0
    canvas = BallBoxVisualizer(bubble_box)
    app.run()
