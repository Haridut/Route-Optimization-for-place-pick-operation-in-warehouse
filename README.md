# Route-Optimization-for-place-pick-operation-in-warehouse

The repository contains my code for route optimization heuristics developed to solve the problem of routing for pick/place operations in a warehouse

Input:  Distance from one location (SKU location) to another
        Possible locations of each SKU
        Order quantity
List of items on oder are randomly generated for now, but can be easily replaced with items you wish to pick/place.

Output: Shortest distance to process the order
	Shortest path to process the order
	Cost of walking in a haphazard way

Distances are pre-computed and just read in as a file for this particular application (A warehouse layout where each rack or location acts as a barrier for traversal)
