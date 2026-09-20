# dev/creator: tubakhxn

import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.trails = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trails[self.next_id] = [centroid]
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.trails[obj_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return {}

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = np.array([self.objects[i] for i in obj_ids])
            input_centroids = np.array(centroids)

            D = np.linalg.norm(obj_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = centroids[col]
                self.disappeared[obj_id] = 0
                self.trails[obj_id].append(centroids[col])
                self.trails[obj_id] = self.trails[obj_id][-20:]
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self._register(centroids[col])

        return dict(self.objects)

    def get_trail(self, obj_id):
        return self.trails.get(obj_id, [])
