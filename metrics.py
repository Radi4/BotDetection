import numpy as np

eps = 0.00001

def dist_point(dist, point):
    def my_dist(x):
        return dist(x, point)
    return my_dist


def min_dist_between_2_clusters(cluster_1, cluster_2, dist, objects):
    distance = 1000000
    for i in cluster_1:
        my_dist = dist_point(dist, objects[i])
        distance = min(distance, np.min(np.apply_along_axis(my_dist, axis = -1, arr = objects[cluster_2])))
    return distance


def max_dist_between_2_clusters(cluster_1, cluster_2, dist, objects):
    distance = 0
    for i in cluster_1:
        my_dist = dist_point(dist, objects[i])
        distance = max(distance, np.max(np.apply_along_axis(my_dist, axis = -1, arr = objects[cluster_2])))
    return distance


#great better
def dann_index(clusters, dist, objects):
    min_dist = 1000000
    max_diameter = 0

    for i in range(len(clusters) - 1):
        if len(clusters[1]) == 1:
            continue

        max_diameter = max(max_diameter, max_dist_between_2_clusters(clusters[i], clusters[i], dist, objects))
        for j in range(i + 1, len(clusters)):
            if len(clusters[j]) == 1:
                continue
            min_dist = min(min_dist, min_dist_between_2_clusters(clusters[i], clusters[j], dist, objects))
    
    return min_dist / (max_diameter + eps)


def std_of_set(X, center):
    return np.mean(np.power(X - center, 2), axis = 0)


def calc_density(cluster, center, dist, stdev):
    my_dist = dist_point(dist, center)
    diff = np.apply_along_axis(my_dist, axis = -1, arr = cluster)
    return np.sum(diff < stdev)


def calc_density_2(cluster_1, cluster_2, mid_point, dist, stdev):
    density = calc_density(cluster_1, mid_point, dist, stdev)
    density += calc_density(cluster_2, mid_point, dist, stdev)
    return density


#less better
def sd_index(clusters, means, dist, objects, alpha = 1):
    stds = np.empty(shape = (len(clusters), objects.shape[1]))
    for i in range(len(clusters)):
        stds[i] = std_of_set(objects[clusters[i]], means[i])
    std_of_all = std_of_set(objects, np.mean(objects, axis = 0))
    
    scatt = np.mean(np.linalg.norm(stds, axis = 1))
    scatt /= np.linalg.norm(std_of_all)
    
    mmax = 0
    mmin = np.inf
    dists = 0
    for index, center in enumerate(means):
        my_dist = dist_point(dist, center)
        diff = np.apply_along_axis(my_dist, axis = -1, arr = means)
        dists += np.power(np.sum(diff), -1)
        mmax = max(mmax, np.max(diff))
        buf = np.delete(diff, index)
        if len(buf) == 0:
            buf = [0]
        mmin = min(mmin, np.min(buf))
    dists /= (mmin + eps)
    dists *= mmax
    return scatt * alpha + dists


#less better
def sdwb_index(clusters, means, dist, objects):
    if len(clusters) <= 1:
        return 100000
    stds = np.empty(shape = (len(clusters), objects.shape[1]))
    for i in range(len(clusters)):
        stds[i] = std_of_set(objects[clusters[i]], means[i])
    std_of_all = std_of_set(objects, np.mean(objects, axis = 0))
    
    scatt = np.mean(np.linalg.norm(stds, axis = 1))
    scatt /= (np.linalg.norm(std_of_all) + eps)
    stdev = np.mean(np.sqrt(np.linalg.norm(stds, axis = 1)))
                    
    density = np.empty(shape = (len(clusters), len(clusters))) 
    dens_bw = 0
    for i in range(means.shape[0]):
        if len(clusters[i]) == 1:
            continue
        density[i][i] = calc_density(objects[clusters[i]], means[i], dist, stdev)
        
        for j in range(i):
            if len(clusters[j]) == 1:
                continue
            dens_bw += density[j][i] / max(density[j][j], density[i][i])
        
        for j in range(i + 1, len(clusters)):
            if len(clusters[j]) == 1:
                continue
            density[j][i] = calc_density_2(objects[clusters[i]], objects[clusters[j]],\
                                            (means[i] + means[j]) * 0.5, dist, stdev)


    dens_bw /= len(clusters)
    dens_bw /= len(clusters) - 1
    return scatt + dens_bw

#great better
def silhouette_index(clusters, dist, objects):
    objects_vals = [None] * len(objects)
    for my_index, my_cluster in enumerate(clusters):
        for i in my_cluster:
            x = objects[i]
            my_dist = dist_point(dist, x)
            objects_vals[i] = [None, np.inf]

            for not_my_index, not_my_cluster in enumerate(clusters):
                dist_sum = np.sum(np.apply_along_axis(my_dist, axis = -1, arr = objects[not_my_cluster]))
                
                if not_my_index == my_index:
                    objects_vals[i][0] = dist_sum / (len(my_cluster) - 1)
                else:
                    dist_sum /= len(not_my_cluster)
                    objects_vals[i][1] = min(dist_sum, objects_vals[i][1])

            objects_vals[i] = (objects_vals[i][1] - objects_vals[i][0]) / max(objects_vals[i][1], objects_vals[i][0])
    
    return sum(objects_vals) / len(objects)


#great better
def simple_silhouette_index(clusters, means, dist, objects):
    objects_vals = [None] * len(objects)
    for index, cluster in enumerate(clusters):
        for i in cluster:
            x = objects[i]
            my_dist = dist_point(dist, x)
            diff = np.apply_along_axis(my_dist, axis = -1, arr = means)
            a = diff[index]
            buf = np.delete(diff, index)
            if len(buf) == 0:
                buf = [0]
            b = np.min(buf)
            objects_vals[i] = (b - a) / max(a, b)
    return sum(objects_vals) / len(objects)


#less better
def cs_index(clusters, means, dist, objects):
    numerator = 0
    for cluster in clusters:
        add = 0
        for x in objects[cluster]:
            my_dist = dist_point(dist, x)
            add += np.max(np.apply_along_axis(my_dist, axis = -1, arr = objects[cluster]))
        add /= len(cluster)
        numerator += add
    
    
    denominator = eps
    for index, center in enumerate(means):
        my_dist = dist_point(dist, center)
        buf = np.delete(np.apply_along_axis(my_dist, axis = -1, arr = means), index)
        if len(buf) == 0:
            buf = [0]
        denominator += np.min(buf)
    
    return numerator / denominator


#less better
def vnnd_index(clusters, dist, objects):
    if len(clusters) <= 1:
        return 100000
    vnnd = 0
    for cluster in clusters:
        means = np.empty(len(cluster))
        for i, x in enumerate(objects[cluster]):
            my_dist = dist_point(dist, x)
            buf = np.delete(np.apply_along_axis(my_dist, axis = -1, arr = objects[cluster]), i)
            if len(buf) == 0:
                buf = [0]
            means[i] = np.min(buf)
        vnnd += np.power(np.std(means, ddof = 1), 2)
    return vnnd


#great better
def score_index(clusters, means, dist, objects):
    if len(clusters) <= 1:
        return 0
    center = np.mean(means, axis = 0)
    my_dist = dist_point(dist, center)
    diff = np.apply_along_axis(my_dist, axis = -1, arr = means)
    diff *= np.array(list(map(len, clusters)))
    return np.sum(diff) / (len(objects) * len(clusters))


#great better
def mb_index(clusters, means, dist, objects, p = 2):
    if len(clusters) <= 1:
        return 0
    e_c = 0
    d = 0
    for index, center in enumerate(means):
        my_dist = dist_point(dist, center)
        e_c += np.sum(np.apply_along_axis(my_dist, axis = -1, arr = objects[clusters[index]]))
        d = max(d, np.max(np.apply_along_axis(my_dist, axis = -1, arr = means)))

    my_dist = dist_point(dist, np.mean(objects, axis = 0))
    e_i = np.sum(np.apply_along_axis(my_dist, axis = -1, arr = objects))
    
    return np.power(e_i * d / (e_c * len(clusters)), p)