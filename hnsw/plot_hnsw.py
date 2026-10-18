import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.colors import to_rgba

from hnsw import HNSW

COLORS = ["#8a96ab", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#ec4899"]


def directed_gradient(X, edges, layer_idx, color, n_steps=10, a_max=1.0, a_min=0.03):
    """
    Каждое направленное ребро (a -> b) режется на n_steps отрезков,
    прозрачность которых падает от a_max у исходящей вершины a до a_min у вершины b.
    Возвращает (segments, rgba) для Line3DCollection.
    """
    src = np.array([X[a] for a, _ in edges])                      # (E, 2)
    dst = np.array([X[b] for _, b in edges])                      # (E, 2)
    t = np.linspace(0, 1, n_steps + 1)
    pts = src[:, None, :] + (dst - src)[:, None, :] * t[None, :, None]    # (E, n+1, 2)
    z = np.full(pts.shape[:2] + (1,), layer_idx, dtype=float)
    pts3 = np.concatenate([pts, z], axis=2)                       # (E, n+1, 3)
    segs = np.stack([pts3[:, :-1], pts3[:, 1:]], axis=2).reshape(-1, 2, 3)

    rgba = np.tile(to_rgba(color), (n_steps, 1))
    rgba[:, 3] = np.linspace(a_max, a_min, n_steps)               # градиент от источника
    return segs, np.tile(rgba, (len(edges), 1))


def plot_hnsw_3d(h, X, ax=None, show_vertical=True):
    """
    Рисует многослойный граф HNSW в 3D.
    Точки лежат в плоскости (x, z) данных, вертикальная ось — номер слоя.
    (В matplotlib вертикальная ось называется z, поэтому данные идут как X и Y, а слой — как Z.)

    h — построенный индекс HNSW, X — массив координат формы (N, 2).
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")

    n_layers = len(h.g)

    # максимальный слой каждой точки
    level = np.zeros(len(X), dtype=int)
    for l, layer in enumerate(h.g):
        for node in layer:
            level[node] = l

    for l, layer in enumerate(h.g):
        color = COLORS[l % len(COLORS)]

        # направленные рёбра слоя: a -> b, если b есть в списке соседей a
        edges = [(a, b) for a, nb in layer.items() for b in nb]
        if edges:
            segs, rgba = directed_gradient(X, edges, l, color,
                                           a_max=0.55 if l == 0 else 1.0)
            ax.add_collection3d(Line3DCollection(
                segs, colors=rgba, linewidths=0.7 if l == 0 else 1.8))

        # сколько рёбер односторонних (обратного ребра b -> a нет)
        es = set(edges)
        one_way = sum((b, a) not in es for a, b in es)

        # точки слоя
        nodes = np.array(sorted(layer))
        ax.scatter(X[nodes, 0], X[nodes, 1], np.full(len(nodes), l),
                   s=8 if l == 0 else 28, color=color, edgecolors="k", linewidths=0.3,
                   label=f"слой {l}: {len(nodes)} точек, {len(es)} рёбер ({one_way} односторонних)")

    # вертикальные пунктиры между копиями одной точки
    vsegs = [[(*X[i], 0), (*X[i], level[i])] for i in range(len(X)) if level[i] > 0]
    if show_vertical and vsegs:     # пустой список сегментов роняет add_collection3d
        ax.add_collection3d(Line3DCollection(vsegs, colors="gray", linestyles=":",
                                             linewidths=0.8, alpha=0.5))

    # входная точка
    ax.scatter(*X[h.entry_point], n_layers - 1, marker="D", s=80, color="crimson",
               edgecolors="k", label="входная точка", zorder=10)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("слой")
    ax.set_zticks(range(n_layers))
    ax.set_zlim(-0.1, n_layers - 0.9)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_box_aspect((1.3, 1, 0.9))
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Граф HNSW по слоям (градиент: яркий конец — исходящая вершина)")
    return ax


if __name__ == "__main__":
    np.random.seed(3)
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal([0, 0], 1.0, (75, 2)),
                   rng.normal([6, 3], 1.0, (75, 2))])

    h = HNSW(M=4, efC=40)
    for x in X:
        h.add(x)

    plot_hnsw_3d(h, X)
    plt.tight_layout()
    plt.savefig("hnsw_3d.png", dpi=150)
    plt.show()