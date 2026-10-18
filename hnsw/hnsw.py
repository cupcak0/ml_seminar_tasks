import heapq
import math
import numpy as np


class HNSW:
    def __init__(self, M=16, efC=100):
        self.M = M
        self.Mmax0 =  2 * M
        self.efC = efC
        self.mL = 1 / math.log(M)
        self.X : np.ndarray = None # type: ignore   
        self.n = 0
        self.g = []     # g[слой][узел] = список соседей
        self.entry_point = None
 
    def _d(self, q, ids):
        """Квадраты расстояний от q до набора точек"""
        return ((self.X[ids] - q) ** 2).sum(1)
 
    def _search_layer(self, q, entry_points, ef, layer):
        visited = set(entry_points)
        d0 = self._d(q, entry_points)
        cand = list(zip(d0, entry_points)); heapq.heapify(cand)                # мин-куча
        W : list[tuple[float, int]] = [(-d, e) for d, e in zip(d0, entry_points)]; heapq.heapify(W)      # макс-куча
        while cand:
            d, c = heapq.heappop(cand)
            if d > -W[0][0]:
                break
            nb = [e for e in self.g[layer][c] if e not in visited]
            visited.update(nb)
            for e, de in zip(nb, self._d(q, nb) if nb else []):
                if len(W) < ef or de < -W[0][0]:
                    heapq.heappush(cand, (de, e))
                    heapq.heappush(W, (-de, e))
                    if len(W) > ef:
                        heapq.heappop(W)
        return sorted((-d, e) for d, e in W)
 
    def _select(self, ids, dq, M):
        """
        param ids: точки из которых мы выбираем
        param dq: квадраты расстояний до точек
        """
        o = np.argsort(dq)
        ids, dq = np.asarray(ids)[o], np.asarray(dq)[o]
        V = self.X[ids]
        s = (V ** 2).sum(1)
        D = s[:, None] + s[None] - 2 * V @ V.T                        # попарные расстояния
        result = []
        for j in range(len(ids)):
            if len(result) == M:
                break
            if (D[j, result] >= dq[j]).all():
                result.append(j)
        result += [j for j in range(len(ids)) if j not in result][:M - len(result)]   # добор
        return ids[result].tolist()
 
    def add(self, v):
        v = np.asarray(v, float)
        if self.X is None:
            self.X = np.empty((16, len(v)))
        elif self.n == len(self.X):
            self.X = np.vstack([self.X, np.empty_like(self.X)])
        i, self.n = self.n, self.n + 1
        self.X[i] = v
 
        l = int(-math.log(1 - np.random.random()) * self.mL) # слой для вставки точки (экспоненциальное распределение)
        top = len(self.g) - 1
        self.g += [{} for _ in range(l - top)]
        for layer in range(l + 1):
            self.g[layer][i] = []
        if self.entry_point is None:
            self.entry_point = i
            return i
 
        entry_point = [self.entry_point]
        for layer in range(top, l, -1):                                  # жадный спуск
            entry_point = [self._search_layer(v, entry_point, 1, layer)[0][1]]
        for layer in range(min(top, l), -1, -1):
            W = self._search_layer(v, entry_point, self.efC, layer)
            self.g[layer][i] = nb = self._select([e for _, e in W], [d for d, _ in W], self.M)
            mm = self.Mmax0 if layer == 0 else self.M
            for n in nb:
                self.g[layer][n].append(i)
                if len(self.g[layer][n]) > mm:                           # обрезка
                    lst = self.g[layer][n]
                    self.g[layer][n] = self._select(lst, self._d(self.X[n], lst), mm)
            entry_point = [e for _, e in W]
        if l > top:
            self.entry_point = i
        return i
 
    def search(self, q, k=5, ef=50):
        q, entry_point = np.asarray(q, float), [self.entry_point]
        for layer in range(len(self.g) - 1, 0, -1):
            entry_point = [self._search_layer(q, entry_point, 1, layer)[0][1]]
        W = self._search_layer(q, entry_point, max(ef, k), 0)[:k]
        return [(math.sqrt(d), e) for d, e in W]                      # (расстояние, номер)
    