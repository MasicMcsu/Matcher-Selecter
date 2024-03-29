/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_GRAPH  
#define SEGMENT_GRAPH  

#include <algorithm>  
#include <cmath>  
#include "disjoint-set.h"  

// threshold function  
#define THRESHOLD(size, c) (c/size)  

typedef struct {
	//float w;  
	double w;
	int a, b;
} edge;

bool operator<(const edge& a, const edge& b);

bool cmpEdge(const edge& edge1, const edge& edge2);
/*
* Segment a graph
*
* Returns a disjoint-set forest representing the segmentation.
*
* num_vertices: number of vertices in graph.
* num_edges: number of edges in graph
* edges: array of edges.
* c: constant for treshold function.
*/
universe* segment_graph(int num_vertices, int num_edges, edge* edges,
	float c);
void segment_graph(int num_vertices, int num_edges, edge* edges,
	float c, universe* u);

#endif 