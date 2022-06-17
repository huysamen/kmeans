package kmeans

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

type DistanceFn[T any] func(a T, b T) float64
type MeanFn[T any] func(n []T) T
type EqualsFn[T any] func(a T, b T) bool

type KMeans[T any] struct {
	data     []T
	distance DistanceFn[T]
	mean     MeanFn[T]
	equals   EqualsFn[T]
	rnd      *rand.Rand
}

type Cluster[T any] struct {
	Centroid T
	Nodes    []T
}

func NewKMeans[T any](data []T, distance DistanceFn[T], mean MeanFn[T], equals EqualsFn[T]) (*KMeans[T], error) {
	if len(data) == 0 {
		return nil, errors.New("no observations provided")
	}

	return &KMeans[T]{
		data:     data,
		distance: distance,
		mean:     mean,
		equals:   equals,
		rnd:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

func (k *KMeans[T]) Calculate(numClusters uint) []Cluster[T] {
	clusters := k.initClusters(numClusters)

	k.partition(clusters)

	return clusters
}

func (k *KMeans[T]) initClusters(numClusters uint) []Cluster[T] {
	var clusters []Cluster[T]

	// 1 -	Choose one center uniformly at random among the data points.
	idx := []int{k.rnd.Intn(len(k.data))}
	clusters = append(clusters, Cluster[T]{Centroid: k.data[idx[0]]})

	// 2 -	For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that
	//		has already been chosen.
	for uint(len(clusters)) < numClusters {
		nodeDistance := -1.0
		nodeIdx := -1

		for i := 0; i < len(k.data); i++ {
			if in(i, idx) {
				continue
			}

			nearestCluster := clusters[k.nearestCluster(k.data[i], clusters)]
			nearestClusterDistance := k.distance(k.data[i], nearestCluster.Centroid)

			if nearestClusterDistance > nodeDistance {
				nodeDistance = nearestClusterDistance
				nodeIdx = i
			}
		}

		// 3 -	Choose one new data point at random as a new center, using a weighted probability distribution where a
		//		point x is chosen with probability proportional to D(x)2.
		idx = append(idx, nodeIdx)
		clusters = append(clusters, Cluster[T]{Centroid: k.data[nodeIdx]})
	} // 4 - Repeat Steps 2 and 3 until k centers have been chosen.

	// 5 -	Now that the initial centers have been chosen, proceed using standard k-means clustering.
	return clusters
}

func in(i int, p []int) bool {
	for _, s := range p {
		if s == i {
			return true
		}
	}

	return false
}

func (k *KMeans[T]) nearestCluster(node T, clusters []Cluster[T]) int {
	minDistance := math.MaxFloat64
	idx := -1

	for i := 0; i < len(clusters); i++ {
		clusterDistance := k.distance(node, clusters[i].Centroid)

		if clusterDistance < minDistance {
			minDistance = clusterDistance
			idx = i
		}
	}

	return idx
}

func (k *KMeans[T]) partition(clusters []Cluster[T]) {
	changed := true

	for changed {
		// 1 -	Assignment
		for i := 0; i < len(k.data); i++ {
			nearestClusterIdx := k.nearestCluster(k.data[i], clusters)
			clusters[nearestClusterIdx].Nodes = append(clusters[nearestClusterIdx].Nodes, k.data[i])
		}

		changedInIteration := false

		// 2 -	Update
		for i := 0; i < len(clusters); i++ {
			pc := clusters[i].Centroid
			clusters[i].Centroid = k.mean(clusters[i].Nodes)
			changedInIteration = changedInIteration || k.equals(pc, clusters[i].Centroid)
		}

		changed = changedInIteration
	}
}
