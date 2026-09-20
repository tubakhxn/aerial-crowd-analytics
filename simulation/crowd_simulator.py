# dev/creator: tubakhxn

import numpy as np

class SimPerson:
    __slots__ = ("id", "x", "y", "vx", "vy", "cluster")

    def __init__(self, pid, x, y, vx, vy, cluster):
        self.id = pid
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.cluster = cluster

class CrowdSimulator:

    def __init__(self, num_people=220, density_bias=0.5, movement_speed=1.2,
                 num_clusters=4, width=1000, height=560, seed=None):
        self.width = width
        self.height = height
        self.num_clusters = max(1, num_clusters)
        self.density_bias = float(np.clip(density_bias, 0.0, 1.0))
        self.movement_speed = movement_speed
        self.rng = np.random.default_rng(seed)
        self.frame_idx = 0
        self._next_id = 0
        self.cluster_centers = self._init_clusters()
        self.people = []
        self.set_population(num_people)

    def _init_clusters(self):
        margin = 0.15
        centers = []
        for _ in range(self.num_clusters):
            cx = self.rng.uniform(margin, 1 - margin) * self.width
            cy = self.rng.uniform(margin, 1 - margin) * self.height
            drift = self.rng.uniform(-0.15, 0.15, size=2)
            centers.append({"x": cx, "y": cy, "dx": drift[0], "dy": drift[1]})
        return centers

    def set_population(self, n):
        n = max(0, int(n))
        if n > len(self.people):
            for _ in range(n - len(self.people)):
                self.people.append(self._spawn_person())
        elif n < len(self.people):
            self.people = self.people[:n]

    def _spawn_person(self):
        cluster = self.rng.integers(0, self.num_clusters)
        c = self.cluster_centers[cluster]
        spread = (1.0 - self.density_bias) * 220 + 40
        x = np.clip(self.rng.normal(c["x"], spread), 0, self.width)
        y = np.clip(self.rng.normal(c["y"], spread), 0, self.height)
        angle = self.rng.uniform(0, 2 * np.pi)
        speed = self.movement_speed * self.rng.uniform(0.4, 1.3)
        vx, vy = np.cos(angle) * speed, np.sin(angle) * speed
        pid = self._next_id
        self._next_id += 1
        return SimPerson(pid, x, y, vx, vy, cluster)

    def step(self):
        self.frame_idx += 1

        for c in self.cluster_centers:
            c["x"] = np.clip(c["x"] + c["dx"], 40, self.width - 40)
            c["y"] = np.clip(c["y"] + c["dy"], 40, self.height - 40)
            if self.rng.random() < 0.02:
                c["dx"] = self.rng.uniform(-0.15, 0.15)
                c["dy"] = self.rng.uniform(-0.15, 0.15)

        for p in self.people:
            c = self.cluster_centers[p.cluster]
            pull = 0.02 * self.density_bias
            p.vx += (c["x"] - p.x) * pull * 0.01 + self.rng.normal(0, 0.05)
            p.vy += (c["y"] - p.y) * pull * 0.01 + self.rng.normal(0, 0.05)
            speed = np.hypot(p.vx, p.vy)
            max_speed = self.movement_speed * 1.6
            if speed > max_speed:
                p.vx, p.vy = p.vx / speed * max_speed, p.vy / speed * max_speed
            p.x = np.clip(p.x + p.vx, 0, self.width)
            p.y = np.clip(p.y + p.vy, 0, self.height)
            if p.x <= 0 or p.x >= self.width:
                p.vx *= -1
            if p.y <= 0 or p.y >= self.height:
                p.vy *= -1

        return self.get_people()

    def get_people(self):
        return [
            {"id": p.id, "x": float(p.x), "y": float(p.y),
             "vx": float(p.vx), "vy": float(p.vy), "cluster": int(p.cluster)}
            for p in self.people
        ]

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.frame_idx = 0
        self._next_id = 0
        self.cluster_centers = self._init_clusters()
        n = len(self.people)
        self.people = []
        self.set_population(n)
