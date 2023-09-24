//
// Created by Clemens Hartmann on 24/08/2023.
//

#ifndef GCN_GRAPH_H
#define GCN_GRAPH_H

#include <cstdlib>
#include <cstddef>

class Graph {
public:
    size_t* index;
    size_t* edges;

    const size_t v_count;
    const size_t e_count;

    Graph(size_t _v_count, size_t _e_count) : v_count(_v_count), e_count(_e_count) {
        index = new size_t[v_count+1] {0};
        edges = new size_t[e_count] {0};
    }
};


#endif //GCN_GRAPH_H
