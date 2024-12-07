import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import lsqr

class VoronoiRBFReconstructor:
    def __init__(self, centers, radii, constraints, values, support_function, max_iter=30, epsilon=1e-4):
        # Split centers and radii into internal (optimized) and external (fixed)
        self.external_centers = torch.tensor(centers[0], dtype=torch.float32, device='cuda').detach()
        self.internal_centers = torch.tensor(centers[1], dtype=torch.float32, device='cuda', requires_grad=True)
        self.external_radii = torch.tensor(radii[0], dtype=torch.float32, device='cuda').detach()
        self.internal_radii = torch.tensor(radii[1], dtype=torch.float32, device='cuda', requires_grad=True)
        
        self.constraints = torch.tensor(constraints, dtype=torch.float32).to('cuda')  # Constraint points on GPU
        self.values = torch.tensor(values, dtype=torch.float32).to('cuda')  # Known function values at the constraint points on GPU
        self.support_function = support_function  # Compact support radial basis function
        self.weights = None  # Initialize weights as None
        self.max_iter = max_iter  # Maximum number of iterations
        self.epsilon = epsilon  # Convergence criterion
        self.optimizer = torch.optim.Adam([self.internal_centers, self.internal_radii], lr=0.1)

    def assemble_matrix(self):
         # Combine external and internal centers and radii
            centers_combined = torch.cat((self.external_centers, self.internal_centers))
            radii_combined = torch.cat((self.external_radii, self.internal_radii))

            # Compute distances between each constraint point and each center
            distances = torch.cdist(self.constraints, centers_combined)
            
            # Determine which constraints are within the support radius of each center
            within_support = distances <= radii_combined.unsqueeze(0)

            # Create sparse matrices for G_t_G and G_t_F
            matrix_size = len(centers_combined)
            G_t_G = lil_matrix((matrix_size, matrix_size), dtype=np.float32)
            G_t_F = np.zeros(matrix_size, dtype=np.float32)

            # Iterate over constraint points and compute contributions to G_t_G and G_t_F
            for k, p_k in enumerate(self.constraints):
                # Find centers that contain the constraint point within their support
                indices = torch.nonzero(within_support[k], as_tuple=False).squeeze()
                if indices.numel() == 0:
                    continue
                
                # Compute support function values for all relevant centers
                phi_values = self.support_function(p_k.unsqueeze(0), centers_combined[indices], radii_combined[indices])
                
                # Update G_t_G and G_t_F in a vectorized manner
                phi_outer = torch.ger(phi_values, phi_values)  # Outer product
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices):
                       G_t_G[idx_i, idx_j] += phi_outer[i, j].cpu().item()
                    G_t_F[idx_i] += (phi_values[i] * self.values[k]).detach().cpu().item()

            return G_t_G.tocsr(), G_t_F

    def solve_weights(self):
        # Assemble the G^T G matrix and the G^T F vector
        G_t_G, G_t_F = self.assemble_matrix()
        
        # Solve G^T G * alpha = G^T F using sparse least squares
        solution = lsqr(G_t_G, G_t_F)
        self.weights = torch.tensor(solution[0], dtype=torch.float32, requires_grad=True).to('cuda')  # Store weights with grad on GPU
        return self.weights

    def reconstruct_function(self, point):
        # Compute the value of the function at a given point
        if self.weights is None:
            self.solve_weights()
        point = torch.tensor(point, dtype=torch.float32).to('cuda')
        centers_combined = torch.cat((self.external_centers, self.internal_centers))
        radii_combined = torch.cat((self.external_radii, self.internal_radii))
        phi_values = self.support_function(point.unsqueeze(0), centers_combined, radii_combined)
        result = torch.sum(self.weights * phi_values)
        return result
    
    def optimize_centers_and_weights(self):
        # Step 1: Initialize weights using the current set of spheres
        self.solve_weights()
        
        iteration = 0
        previous_loss = float('inf')
        
        while iteration < self.max_iter:
            iteration += 1
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Step 2: Compute the reconstruction loss
            loss = self.compute_reconstruction_loss()
            
            # Step 3: Backpropagate the loss
            loss.backward()
            
            # Step 4: Update centers and radii using the optimizer
            self.optimizer.step()
            
            # Step 5: Check for convergence
            reconstruction_loss = loss.item()
            if abs(previous_loss - reconstruction_loss) < self.epsilon:
                print(f"Converged after {iteration} iterations with loss: {reconstruction_loss}")
                print(f"Final internal centers: {self.internal_centers.detach().cpu().numpy()}")
                print(f"Final internal radii: {self.internal_radii.detach().cpu().numpy()}")
                break
            previous_loss = reconstruction_loss
            
            print(f"Iteration {iteration}: Reconstruction loss: {reconstruction_loss}")

    def compute_reconstruction_loss(self):
        # Compute the overall reconstruction loss using vectorized operations
        predicted_values = torch.stack([self.reconstruct_function(p_k) for p_k in self.constraints])
        reconstruction_loss = torch.sum((predicted_values - self.values) ** 2)
        # regularization = torch.sum(self.internal_radii ** 2) * 0.01 + torch.sum(torch.norm(self.internal_centers, dim=1) ** 2) * 0.01
        # distance_regularization = torch.sum(1.0 / (torch.cdist(self.internal_centers, self.internal_centers) + 1e-6)) * 0.01
        # regularization += distance_regularization
        # loss = reconstruction_loss + regularization
        loss = reconstruction_loss 
        return loss

def compactly_supported_rbf(x, c, s):
    r = torch.norm(x - c, p=2, dim=-1) / s  # Compute distance along the last dimension
    mask = r < 1.0
    result = torch.zeros_like(r)
    result[mask] = ((1 - r[mask]) ** 4) * (1 + 4 * r[mask]) * s[mask] if s.ndim > 0 else s
    return result


import time
# Example usage:
# centers = [
#     np.random.uniform(-5, 5, (100, 3)),  # External centers (not optimized)
#     np.random.uniform(-5, 5, (100, 3))   # Internal centers (optimized)
# ]
# radii = [
#     np.random.uniform(0.5, 2.5, 100),  # External radii (not optimized)
#     np.random.uniform(0.5, 2.5, 100)   # Internal radii (optimized)
# ]
# constraints = np.random.uniform(-5, 5, (6000, 3))  # Constraint points
# values = np.random.uniform(-2, 2, 6000)  # Known function values
# start_time = time.time()
# reconstructor = VoronoiRBFReconstructor(centers, radii, constraints, values, compactly_supported_rbf)

# reconstructor.optimize_centers_and_weights()
# end_time = time.time()
# # print(f"initial internal centers after optimization: {centers[1]}")
# # print(f"initial internal radii after optimization: {radii[1]}")
# # print(f"Updated internal centers after optimization: {reconstructor.internal_centers.detach().cpu().numpy()}")
# # print(f"Updated internal radii after optimization: {reconstructor.internal_radii.detach().cpu().numpy()}")

# # Reconstruct the value of the function at an arbitrary point
# print(f"Execution time2: {end_time - start_time:.4f} seconds")
# point = np.array([1.0, 1.0, 1.0])
# value_at_point = reconstructor.reconstruct_function(point)
# print("Value at point:", value_at_point)
