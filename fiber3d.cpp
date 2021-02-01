#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <random>
#include "stdlib.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "nr3.h"
#include "mins.h"
#include "mins_ndim.h"
#include <sys/stat.h>

using namespace Eigen;
using namespace std;

// Parameters
const int N = 10;				// network is NxNxN sites
const double p_0 = 1.0;		// probability of adding bond
const double mu_0 = 100000.0; 		// stretching modulus
const int mugr = 5;
const double g_0 = mu_0/pow(10, mugr);	// bending rigidity ("kappa")
const double f_0 = -g_0/1000.0;		// dipole force
string my_name = "output/fiber"+to_string(N)+"p35k"+to_string(mugr)+"r10";

// Constants
const double bond_length = 1.0;			// unstretched bond length (units of distance)
const int D = 3;						// dimension
const int NS = D*N*N*N;				// Total number of DOF for SVD
const double HA = sqrt(3.0)/2.0;		// triangle altitude
const double HC = 1.0/(2.0*sqrt(3.0));	// triangle distance from side to center
const double HT = sqrt(6.0)/3.0;		// height of tetrahedron
const int branched = 1;

class Node {
	/* holds the information of one node on an FCC lattice */
		int i, j, k; 				//node indices
		int index;					//vector index
		double x_pos, y_pos, z_pos; //initial node position on grid
		double dx, dy, dz; 			//node displacement
		int connected[12]; 			//neighbor connectivity
		Node *n[12]; 				//neighbors
		double mu[12]; 				//stretching modulus
		double g[12]; 				//bending modulus
	public:
		Node (int, int, int);
		void connect_neighbor(Node* neighbor, int d) {
			n[d] = neighbor;
			connected[d] = 1;
		}
		void disconnect_neighbor(int d) {
			connected[d] = 0;
		}
		void set_disp(double a, double b, double c) {
			dx = a;
			dy = b;
			dz = c;
		}
		void set_mu (double mu_d, int d) {
			mu[d] = mu_d;
		}
		void set_g (double g_d, int d) {
			g[d] = g_d;
		}
		int get_i() { return i; }
		int get_j() { return j; }
		int get_k() { return k; }
		int get_index() { return index; }
		double get_x_pos() { return x_pos; }
		double get_y_pos() { return y_pos; }
		double get_z_pos() { return z_pos; }
		double get_dx() { return dx; }
		double get_dy() { return dy; }
		double get_dz() { return dz; }
		int get_connected(int d) { return connected[d]; }
		Node* get_n(int d) { return n[d]; }
		double get_mu(int d) { return mu[d]; }
		double get_g(int d) { return g[d]; }
};

Node::Node(int my_i, int my_j, int my_k) {
	// initialize a Node, set the initial positions, moduli, connectivity, vector index
	i = my_i;
	j = my_j;
	k = my_k;
	
	x_pos = bond_length * (double(i) + double(j)/2.0 + double(k)/2.0);
	y_pos = bond_length * (double(j)*HA + double(k)*HC);
	z_pos = bond_length * double(k) * HT;
	
	dx = 0.0;
	dy = 0.0;
	dz = 0.0;
	
	for (int d = 0; d < 12; d++) { connected[d] = 0; }
	for (int d = 0; d < 12; d++) { mu[d] = 0.0; g[d] = 0.0; }
	
	index = D * (i + N*j + N*N*k);
}

class Network {
	/* Contains NxNxN lattice and functions to create/act on 3d fiber network */
		Node *nodes[N][N][N];
		int root_node[N*N*N];
		int size[N*N*N];
		
		MatrixXd dmatrix, U, V;
		VectorXd S;
		int rank;
		
	public:
		Network();
		
		int displace(int i1, int di) {
			int i2 = i1 + di;
			if (i2 > N-1) i2 = i2 - N;
			if (i2 < 0) i2 = i2 + N;
			return i2;
		}
		
		VecDoub ref_ij(int d) {
			// returns the r_ij of two nodes in the unprobed reference network
			VecDoub vec(3);
			double mag;
			int di, dj, dk;
			
			switch (d) {
  				case 0:
					di = 1; dj = 0; dk = 0;
					break;
				case 1:
					di = 0; dj = 1; dk = 0;
					break;
				case 2:
					di = -1; dj = 1; dk = 0;
					break;
				case 3:
					di = 0; dj = 0; dk = 1;
					break;
				case 4:
					di = -1; dj = 0; dk = 1;
					break;
				case 5:
					di = 0; dj = -1; dk = 1;
					break;
				case 6:
					di = -1; dj = 0; dk = 0;
					break;
				case 7:
					di = 0; dj = -1; dk = 0;
					break;
				case 8:
					di = 1; dj = -1; dk = 0;
					break;
				case 9:
					di = 0; dj = 0; dk = -1;
					break;
				case 10:
					di = 1; dj = 0; dk = -1;
					break;
				case 11:
					di = 0; dj = 1; dk = -1;
					break;
				default:
					cout << "Error: check bond number" << endl;
  			}
			
			vec[0] = bond_length * (di + dj/2.0 + dk/2.0);
			vec[1] = bond_length * (dj*HA + dk*HC);
			vec[2] = bond_length * (dk*HT);
			
			mag = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
			
			vec[0] = vec[0] / mag;
			vec[1] = vec[1] / mag;
			vec[2] = vec[2] / mag;
			
			return vec;
		}
		
		VecDoub r_ij(int i1, int j1, int k1, int di, int dj, int dk) {
			// return the vector between two nodes separated by di, dj, dk
			VecDoub vec(3);
			double x_offset, y_offset, z_offset;
			
			int i2 = displace(i1, di);
			int j2 = displace(j1, dj);
			int k2 = displace(k1, dk);
			
			x_offset = bond_length * (double(di) + double(dj)/2.0 + double(dk)/2.0);
			y_offset = bond_length * (double(dj)*HA + double(dk)*HC);
			z_offset = bond_length * (double(dk)*HT);
			
			vec[0] = nodes[i2][j2][k2]->get_dx() - nodes[i1][j1][k1]->get_dx() + x_offset;
			vec[1] = nodes[i2][j2][k2]->get_dy() - nodes[i1][j1][k1]->get_dy() + y_offset;
			vec[2] = nodes[i2][j2][k2]->get_dz() - nodes[i1][j1][k1]->get_dz() + z_offset;
			return vec;
		}
		
		double mag_r_ij(int i1, int j1, int k1, int di, int dj, int dk) {
			VecDoub vec(3);
			vec = this->r_ij(i1, j1, k1, di, dj, dk);
			return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}
		
		void add_bond(int i1, int j1, int k1, int d) {
			/* connect a Node to its neighbor d, accounting for Lees-Edwards boundary conditions
			This assumes the lattice shape is a parallelogram, not square w/ zig-zag sides */
			int di, dj, dk, i2, j2, k2;
			
			switch (d) {
  				case 0:
					di = 1; dj = 0; dk = 0;
					break;
				case 1:
					di = 0; dj = 1; dk = 0;
					break;
				case 2:
					di = -1; dj = 1; dk = 0;
					break;
				case 3:
					di = 0; dj = 0; dk = 1;
					break;
				case 4:
					di = -1; dj = 0; dk = 1;
					break;
				case 5:
					di = 0; dj = -1; dk = 1;
					break;
				case 6:
					di = -1; dj = 0; dk = 0;
					break;
				case 7:
					di = 0; dj = -1; dk = 0;
					break;
				case 8:
					di = 1; dj = -1; dk = 0;
					break;
				case 9:
					di = 0; dj = 0; dk = -1;
					break;
				case 10:
					di = 1; dj = 0; dk = -1;
					break;
				case 11:
					di = 0; dj = 1; dk = -1;
					break;
				default:
					cout << "Error: check bond number" << endl;
					
  			}
			
			i2 = displace(i1, di);
			j2 = displace(j1, dj);
			k2 = displace(k1, dk);
			
			nodes[i1][j1][k1]->connect_neighbor(nodes[i2][j2][k2], d);
			nodes[i2][j2][k2]->connect_neighbor(nodes[i1][j1][k1], d+6);
			my_union(i1, j1, k1, i2, j2, k2);
		}
		
		void set_bonds(double p) {
		/* connect each node to its neighbors with probability p, keep largest connected segment */
			int index;
			int i2, j2, k2, di, dj, dk;
			int max_size = 0;
			int max_root;
			double r;
			
			// initialize
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						index = nodes[i][j][k]->get_index()/D;
						root_node[index] = index;
						size[index] = 1;
						for (int d = 0; d != 12; d++) {
							nodes[i][j][k]->disconnect_neighbor(d);
						}
					}
				}
			}
			
			// add the bonds
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						for (int d = 0; d != 6; d++) {		//entire fcc lattice
							r = rand()/static_cast<double>(RAND_MAX);
							if (r < p) {
								add_bond(i, j, k, d);
							}
						}
					}
				}
			}
			
			// find the parent of the largest connected component
			for (int i = 0; i != N*N*N; i++) {
				if (find(i) == i) {
					if (size[i] > max_size) {
						max_size = size[i];
						max_root = find(i);
					}
				}
			}
			
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						index = nodes[i][j][k]->get_index()/D;
						if (find(index) != max_root) {
							for (int d = 0; d != 12; d++) {
								nodes[i][j][k]->disconnect_neighbor(d);
							}
						}
					}
				}
			}
		}
		
		int find(int u) {
			//returns the parent (root node) of u
			while (root_node[u] != u) {
				u = root_node[u];
			}
			return u;
		}
		
		void my_union(int i1, int j1, int k1, int i2, int j2, int k2) {
			int index;
			index = nodes[i1][j1][k1]->get_index()/D;
			int a = index;
			index = nodes[i2][j2][k2]->get_index()/D;
			int b = index;
			int r1 = find(a);
			int r2 = find(b);
			int dr;
			if (r1 != r2) {
				if (size[a] < size[b]) {
					dr = r2;
					r2 = r1;
					r1 = dr;
				}
				root_node[r2] = r1;
				size[r1] = size[r1] + size[r2];
			}
		}
		
		void set_uniform_mu() {
			//loop through nodes, assign mu_0 to bonds 0 through 12
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						for (int d = 0; d != 12; d++) {
							nodes[i][j][k]->set_mu(mu_0, d);
						}
					}
				}
			}
		}
		
		void set_uniform_g(double g) {
			//loop through nodes, assign g0 to each existing (coaxial) bonds
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						for (int d = 0; d != 12; d++) {
							nodes[i][j][k]->set_g(g, d);
						}
					}
				}
			}
		}
		
		void fill_dmatrix() {
			int i2, j2, k2;
			int row = 0;
			
			for (int i = 0; i != NS; i++) {
				for (int j = 0; j != NS; j++) {
					//cout << i << " " << j << endl;
					dmatrix(i, j) = 0;
				}
			}
			
			// add stretching and bending to branched fiber model
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						if (branched == 0) {
							for (int d = 0; d != 6; d++) {
								// add stretching
								if (nodes[i][j][k]->get_connected(d)) {
									this->add_stretching(row, i, j, k, d);
								}
						
								if (nodes[i][j][k]->get_connected(d+6)) {
									this->add_stretching(row, i, j, k, d+6);
								}
							
								// coaxial fiber bending
								if (nodes[i][j][k]->get_connected(d) and nodes[i][j][k]->get_connected(d+6)) {
									this->add_bending_ijk(row, i, j, k, d, d+6);
								}
							
								// next nearest coaxial
								if ((nodes[i][j][k]->get_connected(d)) and (nodes[i][j][k]->get_n(d)->get_connected(d))) {
									i2 = nodes[i][j][k]->get_n(d)->get_i();
									j2 = nodes[i][j][k]->get_n(d)->get_j();
									k2 = nodes[i][j][k]->get_n(d)->get_k();
									this->add_bending_jkl(row, i2, j2, k2, d, d+6);
								}
							
								if ((nodes[i][j][k]->get_connected(d+6)) and (nodes[i][j][k]->get_n(d+6)->get_connected(d+6))) {
									i2 = nodes[i][j][k]->get_n(d+6)->get_i();
									j2 = nodes[i][j][k]->get_n(d+6)->get_j();
									k2 = nodes[i][j][k]->get_n(d+6)->get_k();
									this->add_bending_jkl(row, i2, j2, k2, d+6, d);
								}
							}
						}
						else if (branched == 1) {
							for (int d = 0; d != 12; d++) {
								// add stretching
								if (nodes[i][j][k]->get_connected(d)) {
									this->add_stretching(row, i, j, k, d);
								}
								
								// add bending to nearest neighbor pairs
								for (int da = 0; da != 12; da++) {
									if (nodes[i][j][k]->get_connected(da)) {
										//now, loop through the remaining un-added arms
										for (int db = 0; db != da; db++) {
											if (nodes[i][j][k]->get_connected(db)) {
												this->add_bending_ijk(row, i, j, k, da, db);
											}
										}
									}
								}
								
								// add bending to first 6 neighboring nodes / next nearest neighbor
								for (int d = 0; d != 6; d++) {
									if (nodes[i][j][k]->get_connected(d)) {
										i2 = nodes[i][j][k]->get_n(d)->get_i();
										j2 = nodes[i][j][k]->get_n(d)->get_j();
										k2 = nodes[i][j][k]->get_n(d)->get_k();
										
										for (int db = 0; db != 12; db++) {
											if (db != (d+6)) {
												if (nodes[i2][j2][k2]->get_connected(db)) {
													this->add_bending_jkl(row, i2, j2, k2, (d+6), db);
												}
											}
										}
									}
								}
								
								// add bending to second 6 neighboring nodes / next nearest neighbor
								for (int d = 6; d != 12; d++) {
									if (nodes[i][j][k]->get_connected(d)) {
										i2 = nodes[i][j][k]->get_n(d)->get_i();
										j2 = nodes[i][j][k]->get_n(d)->get_j();
										k2 = nodes[i][j][k]->get_n(d)->get_k();
										
										for (int db = 0; db != 12; db++) {
											if (db != (d-6)) {
												if (nodes[i2][j2][k2]->get_connected(db)) {
													this->add_bending_jkl(row, i2, j2, k2, (d-6), db);
												}
											}
										}
									}
								}
								
								
							}
						}
						
						row = row + D;
					}
				}
			}
		}
		
		void add_stretching(int row, int i1, int j1, int k1, int d) {
			double uix = nodes[i1][j1][k1]->get_dx();	//i
			double uiy = nodes[i1][j1][k1]->get_dy();
			double uiz = nodes[i1][j1][k1]->get_dz();
			
			double ujx = nodes[i1][j1][k1]->get_n(d)->get_dx();	//j
			double ujy = nodes[i1][j1][k1]->get_n(d)->get_dy();
			double ujz = nodes[i1][j1][k1]->get_n(d)->get_dz();
			
			double rijx = ref_ij(d)[0];
			double rijy = ref_ij(d)[1];
			double rijz = ref_ij(d)[2];
			
			double s = (-1.0/2.0) * nodes[i1][j1][k1]->get_mu(d) / bond_length;
			
			int dof;
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*pow(rijx,2));
			dmatrix(row, dof+1) += s*(2*rijx*rijy);
			dmatrix(row, dof+2) += s*(2*rijx*rijz);
			
			dof = nodes[i1][j1][k1]->get_n(d)->get_index();
			dmatrix(row, dof) += s*(-2*pow(rijx,2));
			dmatrix(row, dof+1) += s*(-2*rijx*rijy);
			dmatrix(row, dof+2) += s*(-2*rijx*rijz);
			
			row = row + 1;
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*rijx*rijy);
			dmatrix(row, dof+1) += s*(2*pow(rijy,2));
			dmatrix(row, dof+2) += s*(2*rijy*rijz);
			
			dof = nodes[i1][j1][k1]->get_n(d)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rijy);
			dmatrix(row, dof+1) += s*(-2*pow(rijy,2));
			dmatrix(row, dof+2) += s*(-2*rijy*rijz);
			
			row = row + 1;
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*rijx*rijz);
			dmatrix(row, dof+1) += s*(2*rijy*rijz);
			dmatrix(row, dof+2) += s*(2*pow(rijz,2));
			
			dof = nodes[i1][j1][k1]->get_n(d)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rijz);
			dmatrix(row, dof+1) += s*(-2*rijy*rijz);
			dmatrix(row, dof+2) += s*(-2*pow(rijz,2));
		}
		
		void add_bending_ijk(int row, int i1, int j1, int k1, int d1, int d2) {
			double s = (-1.0/2.0) * nodes[i1][j1][k1]->get_g(d1) / (bond_length*bond_length*bond_length);
			
			double rjkx = ref_ij(d1)[0];
			double rjky = ref_ij(d1)[1];
			double rjkz = ref_ij(d1)[2];
			
			double rijx = -ref_ij(d2)[0];
			double rijy = -ref_ij(d2)[1];
			double rijz = -ref_ij(d2)[2];
			
			int dof;
			//force balance in the x-direction of force, "row" = row
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*(-2*rijy*(rijy + rjky) - 2*rijz*(rijz + rjkz)); 	//uix
			dmatrix(row, dof+1) += s*2*rijx*(rijy + rjky);							//uiy
			dmatrix(row, dof+2) += s*2*rijx*(rijz + rjkz);							//uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*rijy*(rijy + rjky) + 2*rjky*(rijy + rjky) + 
      		  2*rijz*(rijz + rjkz) + 2*rjkz*(rijz + rjkz));								//ujx
			dmatrix(row, dof+1) += s*(-2*rijx*(rijy + rjky) - 2*rjkx*(rijy + rjky));	//ujy
			dmatrix(row, dof+2) += s*(-2*rijx*(rijz + rjkz) - 2*rjkx*(rijz + rjkz));	//ujz
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*(-2*rjky*(rijy + rjky) - 2*rjkz*(rijz + rjkz));	//ukx
			dmatrix(row, dof+1) += s*2*rjkx*(rijy + rjky); 							//uky
			dmatrix(row, dof+2) += s*2*rjkx*(rijz + rjkz); 							//ukz
			
			row = row + 1;	//force balance in the y-direction of force, "row" = row + 1
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*2*rijy*(rijx + rjkx); 							//uix
			dmatrix(row, dof+1) += s*(-2*rijx*(rijx + rjkx) - 2*rijz*(rijz + rjkz)); //uiy
			dmatrix(row, dof+2) += s*2*rijy*(rijz + rjkz); 							//uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(-2*rijy*(rijx + rjkx) - 2*(rijx + rjkx)*rjky); 	//ujx
			dmatrix(row, dof+1) += s*(2*rijx*(rijx + rjkx) + 2*rjkx*(rijx + rjkx) +
				2*rijz*(rijz + rjkz) + 2*rjkz*(rijz + rjkz)); 						//ujy
			dmatrix(row, dof+2) += s*(-2*rijy*(rijz + rjkz) - 2*rjky*(rijz + rjkz)); //ujz
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*2*(rijx + rjkx)*rjky;								//ukx
			dmatrix(row, dof+1) += s*(-2*rjkx*(rijx + rjkx) - 2*rjkz*(rijz + rjkz)); //uky
			dmatrix(row, dof+2) += s*2*rjky*(rijz + rjkz);							//ukz
			
			row = row + 1;	//force balance in the z-direction of force, "row" = row + 1
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*2*rijz*(rijx + rjkx);								//uix
			dmatrix(row, dof+1) += s*2*rijz*(rijy + rjky);							//uiy
			dmatrix(row, dof+2) += s*(-2*rijx*(rijx + rjkx) - 2*rijy*(rijy + rjky)); //uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(-2*rijz*(rijx + rjkx) - 2*(rijx + rjkx)*rjkz);	//ujx
			dmatrix(row, dof+1) += s*(-2*rijz*(rijy + rjky) - 2*(rijy + rjky)*rjkz); //ujy
			dmatrix(row, dof+2) += s*(2*rijx*(rijx + rjkx) + 2*rjkx*(rijx + rjkx) +
				2*rijy*(rijy + rjky) + 2*rjky*(rijy + rjky));						//ujz
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*2*(rijx + rjkx)*rjkz;								//ukx
			dmatrix(row, dof+1) += s*2*(rijy + rjky)*rjkz;							//uky
			dmatrix(row, dof+2) += s*(-2*rjkx*(rijx + rjkx) - 2*rjky*(rijy + rjky)); //ukz
		}
		
		void add_bending_jkl(int row, int i1, int j1, int k1, int d1, int d2) {
			double s = (-1.0/2.0) * nodes[i1][j1][k1]->get_g(d1) / (bond_length*bond_length*bond_length);
			
			double rjkx = ref_ij(d2)[0];
			double rjky = ref_ij(d2)[1];
			double rjkz = ref_ij(d2)[2];
			
			double rijx = -ref_ij(d1)[0];
			double rijy = -ref_ij(d1)[1];
			double rijz = -ref_ij(d1)[2];
			
			int dof;
			//force balance in the x-direction of force, "row" = row
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*(2*pow(rijy,2) + 2*pow(rijz,2)); 	//uix
			dmatrix(row, dof+1) += s*(-2*rijx*rijy);							//uiy
			dmatrix(row, dof+2) += s*(-2*rijx*rijz);							//uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(-2*pow(rijy,2) - 2*pow(rijz,2) - 2*rijy*rjky - 					2*rijz*rjkz);								//ujx
			dmatrix(row, dof+1) += s*(2*rijx*rijy + 2*rijy*rjkx);	//ujy
			dmatrix(row, dof+2) += s*(2*rijx*rijz + 2*rijz*rjkx);	//ujz
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*(2*rijy*rjky + 2*rijz*rjkz);	//ukx
			dmatrix(row, dof+1) += s*(-2*rijy*rjkx); 							//uky
			dmatrix(row, dof+2) += s*(-2*rijz*rjkx); 							//ukz
			
			row = row + 1;	//force balance in the y-direction of force, "row" = row + 1
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rijy); 							//uix
			dmatrix(row, dof+1) += s*(2*pow(rijx,2) + 2*pow(rijz,2)); //uiy
			dmatrix(row, dof+2) += s*(-2*rijy*rijz); 							//uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*rijx*rijy + 2*rijx*rjky); 	//ujx
			dmatrix(row, dof+1) += s*(-2*pow(rijx,2) - 2*pow(rijz,2) - 2*rijx*rjkx
				- 2*rijz*rjkz); 						//ujy
			dmatrix(row, dof+2) += s*(2*rijy*rijz + 2*rijz*rjky); //ujz
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rjky);								//ukx
			dmatrix(row, dof+1) += s*(2*rijx*rjkx + 2*rijz*rjkz); //uky
			dmatrix(row, dof+2) += s*(-2*rijz*rjky);							//ukz
			
			row = row + 1;	//force balance in the z-direction of force, "row" = row + 1
			
			dof = nodes[i1][j1][k1]->get_n(d1)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rijz);								//uix
			dmatrix(row, dof+1) += s*(-2*rijy*rijz);							//uiy
			dmatrix(row, dof+2) += s*(2*pow(rijx,2) + 2*pow(rijy,2)); //uiz
			
			dof = nodes[i1][j1][k1]->get_index();
			dmatrix(row, dof) += s*(2*rijx*rijz + 2*rijx*rjkz);	//ujx
			dmatrix(row, dof+1) += s*(2*rijy*rijz + 2*rijy*rjkz); //ujy
			dmatrix(row, dof+2) += s*(-2*pow(rijx,2) - 2*pow(rijy,2) - 2*rijx*rjkx
				- 2*rijy*rjky);						//ujz
			
			dof = nodes[i1][j1][k1]->get_n(d2)->get_index();
			dmatrix(row, dof) += s*(-2*rijx*rjkz);								//ukx
			dmatrix(row, dof+1) += s*(-2*rijy*rjkz);							//uky
			dmatrix(row, dof+2) += s*(2*rijx*rjkx + 2*rijy*rjky); //ukz
		}
		
		void decompose() {
			ofstream myfile;
			string myfile_name = my_name+"-sv.txt";
			myfile.open(myfile_name);
			
			ofstream binsvd;
			string myfile_name2 = my_name+"-sv_binary.txt";
			binsvd.open(myfile_name2, ios::binary);
			
			JacobiSVD<MatrixXd> svd(dmatrix, ComputeThinU | ComputeThinV);
		   	
			//svd.setThreshold(1e-13);
			
			S = svd.singularValues();
			U = svd.matrixU();
			V = svd.matrixV();
			
			rank = svd.rank();
			
			cout << "Rank: " << rank << endl;
			myfile << rank << endl;
			binsvd.write(reinterpret_cast<char*>( &rank ), sizeof rank);
			
			cout << "singular values: " << endl;
			for (int q = 0; q < NS; q++) {
				cout << S[q] << " ";
				myfile << setprecision(15) << S[q] << " ";
				binsvd.write(reinterpret_cast<char*>( &S[q] ), sizeof S[q]);
			}
			
			cout << endl;
			myfile << endl;
			
			cout << "SVD complete. Singular values: " << rank << endl;
			
			myfile << "U" << endl;
			for (int j = 0; j < NS; j++) {
				for (int i = 0; i < NS; i++) {
					myfile << setprecision(15) << U(i, j) << " ";
					binsvd.write(reinterpret_cast<char*>( &U(i, j) ), sizeof U(i, j) );
				}
				myfile << endl;
			}
			myfile << "V" << endl;
			for (int j = 0; j < NS; j++) {
				for (int i = 0; i < NS; i++) {
					myfile << setprecision(15) << V(i, j) << " ";
					binsvd.write(reinterpret_cast<char*>( &V(i, j) ), sizeof V(i, j) );
				}
				myfile << endl;
			}
		}
		
		double probe(int i1, int j1, int k1, int di, int dj, int dk, int dipole) {
			double k_eff, initial_disp, final_disp, strain;
			double force = f_0;
			VecDoub f(NS);
			VecDoub x(NS);
			VecDoub ft(rank);
			VecDoub xt(rank);
			int i2 = displace(i1, di);
			int j2 = displace(j1, dj);
			int k2 = displace(k1, dk);
			
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						nodes[i][j][k]->set_disp(0.0, 0.0, 0.0);
					}
				}
			}
			
			initial_disp = this->mag_r_ij(i1, j1, k1, di, dj, dk);
			
			for (int i = 0; i != NS; i++) {
				f[i] = 0.0;
				x[i] = 0.0;
			}
			
			for (int i = 0; i != rank; i++) {
				ft[i] = 0.0;
				xt[i] = 0.0;
			}
			
			int dof;
			dof = nodes[i1][j1][k1]->get_index();
			f[dof] = force*r_ij(i1, j1, k1, di, dj, dk)[0]/mag_r_ij(i1, j1, k1, di, dj, dk);
			f[dof+1] = force*r_ij(i1, j1, k1, di, dj, dk)[1]/mag_r_ij(i1, j1, k1, di, dj, dk);
			f[dof+2] = force*r_ij(i1, j1, k1, di, dj, dk)[2]/mag_r_ij(i1, j1, k1, di, dj, dk);
			if (dipole == 1) {
			dof = nodes[i2][j2][k2]->get_index();
				f[dof] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[0]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
				f[dof+1] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[1]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
				f[dof+2] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[2]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
			}
			
			double sum;
			for (int i = 0; i < rank; i++) {
				sum = 0.0;
				for (int j = 0; j < NS; j++) {
					sum += U(j, i)*f[j];
				}
				ft[i] = sum;
			}
			
			for (int i = 0; i < rank; i++) {
				xt[i] = ft[i] / S[i];
			}
			
			for (int i = 0; i < NS; i++) {
				sum = 0.0;
				//for (int j = 0; j < NS; j++) {
				for (int j = 0; j < rank; j++) {
					sum += V(i, j)*xt[j];
				}
				x[i] = sum;
			}
			
			dof = 0;
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						nodes[i][j][k]->set_disp(x[dof], x[dof+1], x[dof+2]);
						dof = dof + D;
					}
				}
			}
			
			if (dipole == 1) {
				final_disp = mag_r_ij(i1, j1, k1, di, dj, dk);
			} else {
				initial_disp = 0;
				final_disp = -sqrt(nodes[i1][j1][k1]->get_dx()*nodes[i1][j1][k1]->get_dx() +
					nodes[i1][j1][k1]->get_dy()*nodes[i1][j1][k1]->get_dy() +
						nodes[i1][j1][k1]->get_dz()*nodes[i1][j1][k1]->get_dz());
			}
			
			strain = (final_disp - initial_disp);
			k_eff = force / strain;
			
			return k_eff / mu_0;
		}
		
		double project_force(int i1, int j1, int k1, int di, int dj, int dk, int dipole) {
			// use the pseudo-inverse to project initial force vector onto the nullspace,
			// returns the magnitude of the projected vector
			
			double force = f_0;
			VecDoub f(NS);
			int i2 = displace(i1, di);
			int j2 = displace(j1, dj);
			int k2 = displace(k1, dk);
			int dof;
			double sum, diff2;
			
			for (int i = 0; i != NS; i++) {
				f[i] = 0.0;
			}
			
			dof = nodes[i1][j1][k1]->get_index();
			f[dof] = force*r_ij(i1, j1, k1, di, dj, dk)[0]/mag_r_ij(i1, j1, k1, di, dj, dk);
			f[dof+1] = force*r_ij(i1, j1, k1, di, dj, dk)[1]/mag_r_ij(i1, j1, k1, di, dj, dk);
			f[dof+2] = force*r_ij(i1, j1, k1, di, dj, dk)[2]/mag_r_ij(i1, j1, k1, di, dj, dk);
			
			if (dipole == 1) {
				dof = nodes[i2][j2][k2]->get_index();
				f[dof] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[0]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
				f[dof+1] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[1]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
				f[dof+2] = force*r_ij(i2, j2, k2, -di, -dj, -dk)[2]/mag_r_ij(i2, j2, k2, -di, -dj, -dk);
			}
			
			diff2 = 0;
			for (int q = 0; q != NS; q++) {
				sum = 0;
				dof = 0;
				for (int k = 0; k != N; k++) {
					for (int j = 0; j != N; j++) {
						for (int i = 0; i != N; i++) {
							sum += dmatrix(dof, q) * nodes[i][j][k]->get_dx();
							sum += dmatrix(dof+1, q) * nodes[i][j][k]->get_dy();
							sum += dmatrix(dof+2, q) * nodes[i][j][k]->get_dz();
							dof = dof + D;
						}
					}
				}
				
				diff2 += (f[q] - sum)*(f[q] - sum);
			}
			
			return sqrt(diff2) / mu_0;
		}
		/*
		void probe_all_fibre() {
			ofstream myfile;
			string my_dip_name = my_name+"-probes-dipole.txt";
			myfile.open(my_dip_name);
			
			ofstream myfile2;
			string my_mono_name = my_name+"-probes-mono.txt";
			myfile2.open(my_mono_name);
			//ofstream myerr;
			//string myerr_name = my_name+"-project_force.txt";
			//myerr.open(myerr_name);
			
			int di, dj, dk, i2, j2, k2, neighbors1, neighbors2;
			double k_eff, f_null;
			
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						for (int d = 0; d != 1; d++) {
							if (d==0) {
								di = 3;
								dj = -2;
								dk = 2;
							}
							
							i2 = displace(i, di);
							j2 = displace(j, dj);
							k2 = displace(k, dk);
							
							neighbors1 = 0;
							neighbors2 = 0;
							for (int dn = 0; dn != 12; dn++) {
								neighbors1 += nodes[i][j][k]->get_connected(dn);
								neighbors2 += nodes[i2][j2][k2]->get_connected(dn);
							}
							
							if ((neighbors1 != 0) and (neighbors2 != 0)) {
								k_eff = probe(i, j, k, di, dj, dk, 1);
								
								f_null = project_force(i, j, k, di, dj, dk, 1);
								//myerr << f_null << endl;
								
								f_null = log10(f_null);
								myfile << k_eff << " " << f_null << endl;
								//if (f_null < -10) {
								//	myfile << k_eff << " " << 1 << endl;
								//} else {
								//	myfile << k_eff << " " << 0 << endl;
								//}
							}
							
							if ((neighbors1 != 0)) {
								k_eff = probe(i, j, k, di, dj, dk, 0);
								
								f_null = project_force(i, j, k, di, dj, dk, 0);
								//myerr << f_null << endl;
								
								f_null = log10(f_null);
								myfile2 << k_eff << " " << f_null << endl;
							}
						}
					}
				}
			}
		}
		*/
		
		void probe_all_fibre() {
			ofstream mydisp;
			string mydisp_name = my_name+"-disp.txt";
			mydisp.open(mydisp_name);
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						for (int d = 0; d != 6; d++) {
							mydisp << nodes[a][b][c]->get_connected(d) << " ";
						}
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_x_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_y_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_z_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			int di, dj, dk, i2, j2, k2, neighbors1, neighbors2;
			double k_eff, proj;
			
			for (int k = 0; k != N; k++) {
				for (int j = 0; j != N; j++) {
					for (int i = 0; i != N; i++) {
						for (int d = 0; d != 1; d++) {
							if (d==0) {
								di = -1;
								dj = 2;
								dk = 1;
							}
							
							i2 = displace(i, di);
							j2 = displace(j, dj);
							k2 = displace(k, dk);
							
							neighbors1 = 0;
							neighbors2 = 0;
							for (int dn = 0; dn != 12; dn++) {
								neighbors1 += nodes[i][j][k]->get_connected(dn);
								neighbors2 += nodes[i2][j2][k2]->get_connected(dn);
							}
							if ((neighbors1 != 0) and (neighbors2 != 0)) {
								k_eff = probe(i, j, k, di, dj, dk, 1);
								proj = project_force(i, j, k, di, dj, dk, 1);
								
								for (int c = 0; c != N; c++) {
									for (int b = 0; b != N; b++) {
										for (int a = 0; a != N; a++) {
											mydisp << nodes[a][b][c]->get_dx() << " ";
										}
									}
								}
								mydisp << endl;
								
								for (int c = 0; c != N; c++) {
									for (int b = 0; b != N; b++) {
										for (int a = 0; a != N; a++) {
											mydisp << nodes[a][b][c]->get_dy() << " ";
										}
									}
								}
								mydisp << endl;
								
								for (int c = 0; c != N; c++) {
									for (int b = 0; b != N; b++) {
										for (int a = 0; a != N; a++) {
											mydisp << nodes[a][b][c]->get_dz() << " ";
										}
									}
								}
								mydisp << endl;
								
								mydisp << i << " " << j << " " << k << " " << i2 << " " << j2 << " " << k2 << " " << k_eff << " " << proj << endl;
								
							}
						}
					}
				}
			}
			
			//mydisp << endl;
			//myfile << endl;
			//myfile2 << endl;
		}
		
		void probe_indep(int fd) {
			ofstream myfile;
			ofstream myfile2;
			if (fd == 1) {
				string myfile_name = my_name+"-probe_indep_keff.txt";
				//myfile.open(myfile_name, ios::app);
				myfile.open(myfile_name);
			
				
				string myfile_name2 = my_name+"-probe_indep_proj.txt";
				//myfile2.open(myfile_name2, ios::app);
				myfile2.open(myfile_name2);
			}
			else {
				//ofstream myfile;
				string myfile_name = my_name+"-probe_indep_keff.txt";
				myfile.open(myfile_name, ios::app);
				//myfile.open(myfile_name);
			
				//ofstream myfile2;
				string myfile_name2 = my_name+"-probe_indep_proj.txt";
				myfile2.open(myfile_name2, ios::app);
				//myfile2.open(myfile_name2);
			}
			
			int di, dj, dk, i2, j2, k2, neighbors1, neighbors2;
			double k_eff, proj;
			int start, end; //for probing only the center in pinned version
			start = 0;
			end = N;
			
			di = -1;
			dj = 2;
			dk = fd;
			
			myfile << di << " " << dj << " " << dk << endl;
			myfile2 << di << " " << dj << " " << dk << endl;
			
			for (int k = start; k != end; k++) {
				for (int j = start; j != end; j++) {
					for (int i = start; i != end; i++) {
						for (int d = 0; d != 1; d++) {
							if (d==0) {
								di = -1;
								dj = 2;
								dk = fd;
							}
						
							i2 = displace(i, di);
							j2 = displace(j, dj);
							k2 = displace(k, dk);
							neighbors1 = 0;
							neighbors2 = 0;
							for (int dn = 0; dn != 12; dn++) {
								neighbors1 += nodes[i][j][k]->get_connected(dn);
								neighbors2 += nodes[i2][j2][k2]->get_connected(dn);
							}
							if ((neighbors1 != 0) and (neighbors2 != 0)) {
								k_eff = probe(i, j, k, di, dj, dk, 1);
								proj = project_force(i, j, k, di, dj, dk, 1);
								
								myfile << k_eff << " ";
								myfile2 << proj << " ";
							}
						}
					}
				}
			}
			
			myfile << endl;
			myfile2 << endl;
		}
		
		void output_state() {
			ofstream mydisp;
			string mydisp_name = my_name+"-state.txt";
			mydisp.open(mydisp_name);
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						for (int d = 0; d != 6; d++) {
							mydisp << nodes[a][b][c]->get_connected(d) << " ";
						}
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_x_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_y_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_z_pos() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_dx() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_dy() << " ";
					}
				}
			}
			mydisp << endl;
			
			for (int c = 0; c != N; c++) {
				for (int b = 0; b != N; b++) {
					for (int a = 0; a != N; a++) {
						mydisp << nodes[a][b][c]->get_dz() << " ";
					}
				}
			}
			mydisp << endl;
		}
		
		void read_p() {
			string myfile_name = my_name+"-state.txt";
			ifstream myfile (myfile_name);
			
			string line;
			if (myfile.is_open()) {
				getline(myfile, line);
				
				for (int k = 0; k != N; k++) {
					for (int j = 0; j != N; j++) {
						for (int i = 0; i != N; i++) {
							for (int d = 0; d != 12; d++) {
								nodes[i][j][k]->disconnect_neighbor(d);
							}
						}
					}
				}
			
				int b = 0;
				int c;
				for (int k = 0; k != N; k++) {
					for (int j = 0; j != N; j++) {
						for (int i = 0; i != N; i++) {
							for (int d = 0; d != 6; d++) {		//entire fcc lattice
								c = line[b] - '0';
								if (c == 1) {
									add_bond(i, j, k, d);
								}
								b = b + 2;
							}
						}
					}
				}
				
				myfile.close();
			}
			
		}
		
		void read_SVD() {
			string myfile_name = my_name+"-sv_binary.txt";
			ifstream myfile (myfile_name, ios::binary);
			
			string line;
			S = VectorXd::Zero(NS);
			U = MatrixXd::Zero(NS, NS);
			V = MatrixXd::Zero(NS, NS);
			
			if (myfile.is_open()) {
				myfile.read(reinterpret_cast<char*>( &rank ), sizeof rank);
				cout << rank << endl;
				
				for (int q = 0; q < NS; q++) {
					myfile.read(reinterpret_cast<char*>( &S[q] ), sizeof S[q]);
					//cout << S[q] << " ";
				}
				//cout << endl;
				
				for (int j = 0; j < NS; j++) {
					for (int i = 0; i < NS; i++) {
						myfile.read(reinterpret_cast<char*>( &U(i, j) ), sizeof U(i,j) );
						//cout << U(i, j) << " ";
					}
					//cout << endl;
				}
				
				for (int j = 0; j < NS; j++) {
					for (int i = 0; i < NS; i++) {
						myfile.read(reinterpret_cast<char*>( &V(i, j) ), sizeof V(i,j) );
						//cout << V(i, j) << " ";
					}
					//cout << endl;
				}
			}
		}
};

Network::Network() {
	dmatrix = MatrixXd::Zero(NS,NS);
	int index;
	
	for (int k = 0; k != N; k++) {
		for (int j = 0; j != N; j++) {
			for (int i = 0; i != N; i++) {
	    		nodes[i][j][k] = new Node(i, j, k);
				index = nodes[i][j][k]->get_index()/D;
				root_node[index] = index;
				size[index] = 1;
			}
		}
	}
}

int main(int argc, char * argv[]) {
	cout << "\n\n\n\n\n\n\n";
	cout << "Program running\n";
	
	cout << "Initializing network" << endl;
	Network test_network;
	
	cout << "Setting mu" << endl;
	test_network.set_uniform_mu();
	
	int ratio = atoi(argv[1]);
	double g = mu_0 / pow(10, ratio);
	
	string p_string = "0.";
	p_string = p_string+argv[2];
	double my_p = stof(p_string);
	
	string my_index = argv[3];
	cout << ratio << " " << my_p << " " << my_index << endl;
	int mi = atoi(argv[3]);
	
	cout << "Seeding random distribution" << endl;
	srand (mi*time(NULL));
	cout << mi*time(NULL) << endl;
	
	string my_output = "output";
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/k" + argv[1];
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/p"+argv[2];
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/" + my_index;
	mkdir(my_output.c_str(), 0700);
	
	my_name = my_output + "/" + "fcc"+to_string(N);
	
	cout << "Setting g" << endl;
	test_network.set_uniform_g(g);
	
	cout << "Setting bonds" << endl;
	//test_network.set_bonds(my_p);
	//test_network.set_bonds(1.0);
	
	//cout << "Outputting state" << endl;
	//test_network.output_state();
	
	//cout << "Running SVD" << endl;
	//test_network.decompose();

	//double keff;
	//keff = test_network.probe(0, 0, 0, 1, 0, 0, 1);
	//cout << keff << endl;
	
	//cout << "Probing fiber" << endl;
	//test_network.probe_all_fibre();
	
	test_network.read_p();
	test_network.read_SVD();
	
	cout << "Filling dmatrix" << endl;
	test_network.fill_dmatrix();
	
	if (atoi(argv[2]) == 30) {
		for (int i = 1; i != 6; i++) {
			test_network.probe_indep(i);
		}
	}
	
	cout << "Done" << endl;
	return 0;
}
