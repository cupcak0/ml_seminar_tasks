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
        self.entry_point = None # type: ignore
 
    def _d(self, q, ids):
        """Квадраты расстояний от q до набора точек"""
        return ((self.X[ids] - q) ** 2).sum(1)
 
    def _search_closest(self, q, entry_point, layer):
        current = entry_point
        
        while True:
            if len(self.g[layer][current]) == 0:
                break
            
            dist = ((self.X[self.g[layer][current]] - q)**2).sum(1)
            nxt = np.argmin(dist)
            if dist[nxt] > ((self.X[current] - q)**2).sum():
                break
            current = self.g[layer][current][nxt]
        return current
    
    def _search_layer(self, q, entry_point, layer, k):
        ans = []
        nvisited = np.ones(self.n+1, dtype=np.dtypes.BoolDType)
        
        heap = list(zip(((self.X[self.g[layer][entry_point]] - q)**2).sum(1),self.g[layer][entry_point])); heapq.heapify(heap)
        heapq.heappush(heap, (((self.X[entry_point]-q)**2).sum(),entry_point))
        
        for i in range(k):
            if len(heap) == 0:
                break
            d, cur = heapq.heappop(heap)
            ans.append((d, int(cur)))
            
            newpts = self.g[layer][cur]
            newpts = np.array(newpts)[nvisited[newpts]].tolist()
            
            nvisited[newpts] = False
            
            for p in newpts:
                heapq.heappush(heap, (((self.X[p]-q)**2).sum(), p))
                
        return ans

 
    def _select(self, ids, dq, M):
        """
        param ids: точки из которых мы выбираем
        param dq: квадраты расстояний до точек
        """
        
        return np.array(ids)[np.argpartition(dq, min(len(dq)-1,M))[:min(len(dq),M)]].tolist()
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
 
        entry_point = self.entry_point
        for layer in range(top, l, -1):                                  # жадный спуск
            entry_point = self._search_closest(v, entry_point, layer)
        for layer in range(min(top, l), -1, -1):
            W = self._search_layer(v, entry_point, layer, self.efC)
            self.g[layer][i] = nb = self._select([e for _, e in W], [d for d, _ in W], self.M)
            mm = self.Mmax0 if layer == 0 else self.M
            for n in nb:
                self.g[layer][n].append(i)
                if len(self.g[layer][n]) > mm:                           # обрезка
                    lst = self.g[layer][n]
                    self.g[layer][n] = self._select(lst, self._d(self.X[n], lst), mm)
            entry_point = W[0][1]
        if l > top:
            self.entry_point = i
        return i
 
    def search(self, q, k=5, ef=50):
        q, entry_point = np.asarray(q, float), self.entry_point
        for layer in range(len(self.g) - 1, 0, -1):
            entry_point = self._search_closest(q, entry_point, layer)
        W = self._search_layer(q, entry_point, 0, k)
        return [(math.sqrt(d), e) for d, e in W]                      # (расстояние, номер)
    