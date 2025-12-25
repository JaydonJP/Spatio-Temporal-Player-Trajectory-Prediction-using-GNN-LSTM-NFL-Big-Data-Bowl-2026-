import sys
print("Executable:", sys.executable)
# print("Path:", sys.path) 

import os

import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
CONFIG = {
    'train_dir': r'c:\OblivionX\Python\AI ML\nfl-big-data-bowl-2026-prediction\train',
    'test_file': r'c:\OblivionX\Python\AI ML\nfl-big-data-bowl-2026-prediction\test_input.csv',
    'batch_size': 32, # Plays per batch
    'model_save_path': 'st_gat_model.pth',
    'epochs': 1, # Changed to 1 as requested
    'learning_rate': 1e-3,
    'hidden_dim': 64,
    'num_heads': 4,
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

# --- Data Loading & Preprocessing ---

class NFLBigDataDataset(Dataset):
    def __init__(self, data_dir, mode='train', scaler=None, encoders=None):
        self.mode = mode
        self.plays = []
        
        # Load data
        if mode == 'train':
            # Load Weeks 1-18
            # WARNING: This consumes a lot of memory. To optimize, use lazy loading or chunking.
            # strict requirement: "use all the data available"
            input_files = sorted(glob.glob(os.path.join(data_dir, 'input_2023_w*.csv')))
            output_files = sorted(glob.glob(os.path.join(data_dir, 'output_2023_w*.csv')))
            
            print(f"Found {len(input_files)} weeks of data.")
            
            all_input_frames = []
            all_output_frames = []
            
            for i, (in_f, out_f) in enumerate(zip(input_files, output_files)):
                print(f"Loading Week {i+1}...", flush=True)
                df_in = pd.read_csv(in_f)
                df_out = pd.read_csv(out_f)
                
                # Basic cleaning
                df_in['player_weight'] = df_in['player_weight'].fillna(200)
                
                # Append to list
                all_input_frames.append(df_in)
                all_output_frames.append(df_out)
                
            self.input_data = pd.concat(all_input_frames, ignore_index=True)
            self.output_data = pd.concat(all_output_frames, ignore_index=True)
            
            # --- Preprocessing ---
            # 1. Encoders
            self.encoders = {}
            # cat_cols = ['play_direction', 'player_position', 'player_side', 'player_role', 'event']
            # Dynamic check
            potential_cat_cols = ['play_direction', 'player_position', 'player_side', 'player_role', 'event']
            cat_cols = [c for c in potential_cat_cols if c in self.input_data.columns]
            
            print(f"Encoding columns: {cat_cols}", flush=True)
            
            for col in cat_cols:
                le = LabelEncoder()
                # Fill NaNs with 'unknown' for robust encoding
                self.input_data[col] = self.input_data[col].fillna('unknown').astype(str)
                le.fit(self.input_data[col])
                self.encoders[col] = le
                self.input_data[f'{col}_enc'] = le.transform(self.input_data[col])
            
            # 2. Scaler
            self.scaler = StandardScaler()
            num_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'player_weight', 'absolute_yardline_number']
            # Ensure columns exist (some might be missing or named differently, checking common ones)
            existing_num_cols = [c for c in num_cols if c in self.input_data.columns]
            
            self.scaler.fit(self.input_data[existing_num_cols].fillna(0))
            self.input_data[existing_num_cols] = self.scaler.transform(self.input_data[existing_num_cols].fillna(0))
            self.num_feature_cols = existing_num_cols
            
            # 3. Grouping
            # Create a lookup for plays: (game_id, play_id) -> data index
            # This is faster than groupby for every item if we pre-calculate indices
            self.play_indices = self.input_data.groupby(['game_id', 'play_id']).indices
            self.play_ids = list(self.play_indices.keys())
            
            # Output lookup
            self.output_play_indices = self.output_data.groupby(['game_id', 'play_id']).indices
            
        else:
            # Test mode not fully implemented in this snippet for brevity, focusing on training
            pass

    def __len__(self):
        return len(self.play_ids)

    def __getitem__(self, idx):
        game_id, play_id = self.play_ids[idx]
        
        # Get indices
        in_indices = self.play_indices[(game_id, play_id)]
        out_indices = self.output_play_indices.get((game_id, play_id), [])

        # Extract data
        play_in = self.input_data.iloc[in_indices]
        play_out = self.output_data.iloc[out_indices]
        
        # Organize by frame and player
        # We need a temporal graph: Nodes = Players, Edges = interactions
        # Input features: [Time, Players, Features]
        
        frames = sorted(play_in['frame_id'].unique())
        players = sorted(play_in['nfl_id'].unique())
        
        num_frames = len(frames)
        num_players = len(players)
        
        # Player ID to index map for this play
        pid_to_idx = {pid: i for i, pid in enumerate(players)}
        
        # Feature Matrix Construction
        # We need to build a tensor of shape [Num_Frames * Num_Players, Feature_Dim] 
        # But for Torch Geometric Temporal interaction, usually [Num_Nodes, Node_Feature_Dim] per frame or [Num_Nodes, Time, Features]
        
        # Let's create a single large graph per play where temporal edges connect frames? 
        # No, better: ST-GNN approach. 
        # 1. Spatial Graph per frame.
        # 2. LSTM over the sequence of graph embeddings.
        
        feature_dim = len(self.num_feature_cols) + len(self.encoders)
        
        # Prepare tensor containers
        # Size: [Num_Frames, Num_Players, Feature_Dim]
        x_tensor = torch.zeros((num_frames, num_players, feature_dim), dtype=torch.float32)
        
        for t, frame in enumerate(frames):
            frame_data = play_in[play_in['frame_id'] == frame]
            
            for _, row in frame_data.iterrows():
                pid = row['nfl_id']
                if pid in pid_to_idx:
                    p_idx = pid_to_idx[pid]
                    
                    # Numerical features
                    feat_vals = row[self.num_feature_cols].values.astype(np.float32)
                    
                    # Categorical features
                    cat_vals = [row[f'{c}_enc'] for c in self.encoders.keys()]
                    
                    # Combine
                    full_feat = np.concatenate([feat_vals, cat_vals])
                    x_tensor[t, p_idx] = torch.tensor(full_feat, dtype=torch.float32)

        # Edge Index (Fully Connected Spatial Graph)
        # Create edges between all players in a frame. 
        # Loop: 0->0, 0->1, ... N->N (Self loops included usually for GCN, GAT can handle it)
        source = []
        target = []
        for i in range(num_players):
            for j in range(num_players):
                source.append(i)
                target.append(j)
        edge_index = torch.tensor([source, target], dtype=torch.long)
        
        # Targets
        # We want to predict x, y for the output frames.
        # Output frames might differ in count.
        # For simplicity in this robust model, we predict the next N frames or specifically the frames in output.
        # The prompt implies predicting trajectory.
        
        # Sort output by frame
        play_out = play_out.sort_values('frame_id')
        out_frames = sorted(play_out['frame_id'].unique())
        
        # Target Tensor: [Num_Output_Frames, Num_Players, 2] (x, y)
        y_tensor = torch.zeros((len(out_frames), num_players, 2), dtype=torch.float32)
        
        for t, frame in enumerate(out_frames):
            frame_data = play_out[play_out['frame_id'] == frame]
            for _, row in frame_data.iterrows():
                pid = row['nfl_id']
                if pid in pid_to_idx:
                    p_idx = pid_to_idx[pid]
                    y_tensor[t, p_idx] = torch.tensor([row['x'], row['y']], dtype=torch.float32)
                    
        # Return PyG Data Object is tricky with temporal batching. 
        # We will return standard PyTorch tensors and handle Graph conversion in the collate or model forward.
        return {
            'x': x_tensor, # [T, N, F]
            'edge_index': edge_index, # [2, N*N]
            'y': y_tensor, # [T_out, N, 2]
            'num_players': num_players,
            'num_frames': num_frames
        }

def custom_collate(batch):
    # Determine max shape for padding
    max_players = max([b['num_players'] for b in batch])
    max_in_frames = max([b['x'].shape[0] for b in batch])
    max_out_frames = max([b['y'].shape[0] for b in batch])
    feature_dim = batch[0]['x'].shape[2]
    
    # Pad and batch
    batch_x = torch.zeros((len(batch), max_in_frames, max_players, feature_dim))
    batch_y = torch.zeros((len(batch), max_out_frames, max_players, 2))
    batch_adj = [] # List of edge_indices adjusted for batching? No, GAT usually takes one huge graph or we process individually.
    
    # For GNN batching, standard is: combine all nodes into one giant graph.
    # But here we have [Batch, Time, Nodes].
    # We will process [Batch*Time] snapshots if we want pure GNN, or use a model that handles loose batches.
    # To keep it efficient: We will use a mask for padding.
    
    mask = torch.zeros((len(batch), max_players), dtype=torch.bool)
    
    for i, b in enumerate(batch):
        p = b['num_players']
        t_in = b['x'].shape[0]
        t_out = b['y'].shape[0]
        
        batch_x[i, :t_in, :p, :] = b['x']
        batch_y[i, :t_out, :p, :] = b['y']
        mask[i, :p] = 1

    return {
        'x': batch_x,      # [B, T, N, F]
        'y': batch_y,      # [B, T_out, N, 2]
        'mask': mask       # [B, N] - valid players
    }

# --- ST-GAT Model ---

class STGATModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads, dropout=0.1):
        super(STGATModel, self).__init__()
        
        self.input_emb = nn.Linear(in_channels, hidden_dim)
        
        # Spatial Attention (GATv2)
        # We will apply this to each frame independently (shared weights)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        
        # Temporal (LSTM)
        # Input: [Batch * Nodes, Time, Hidden]
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2) # X, Y
        )
        
    def forward(self, x, mask):
        # x: [B, T, N, F]
        B, T, N, F_dim = x.shape
        
        # Flatten batch and time for GNN
        # We need efficient GNN processing.
        # Treat every (Batch, Frame) as a separate graph.
        # Nodes = B * T * N
        
        # 1. Feature Embedding
        x_emb = self.input_emb(x.view(-1, F_dim)) # [B*T*N, Hidden]
        
        # 2. Build Edge Index dynamically or iterate
        # Building a giant edge index for B*T graphs is expensive and memory heavy.
        # Optimization: Apply GAT only on active nodes?
        # Or simpler: Iterate over Time if T is small (~50 frames). B*T is large (32*50=1600).
        # Let's pivot: Apply GAT frame-by-frame on [B*N] nodes? 
        #   Yes, assume B*N nodes, but edges only exist within B groups.
        
        # Construct Block Diagonal Adjacency for the Batch (B graphs of N nodes)
        # Since N changes per play (padded), we use the Mask.
        
        # Actually, standard PyG batching handles this best.
        # But our data loader returns dense tensors.
        # Let's reshape to [B*T, N, F] and treat as B*T independent graphs.
        
        # Fast approach for demonstration: Use GCN/GAT on fully connected N nodes.
        # We can implement a dense GAT if N is small (22).
        # PyG has DenseGATConv! Perfect for [B, N, F].
        # But we have [B*T, N, F].
        
        # Let's iterate over T steps, pass [B, N, F] to DenseGAT.
        
        pass 
        # Re-defining structure to use DenseGAT for speed on [B, N]
        
    def forward_dense(self, x, mask):
        # x: [B, T, N, F]
        B, T, N, _ = x.shape
        
        # Flatten time into batch for spatial processing
        x_flat = x.view(B*T, N, -1) # [B*T, N, F]
        x_emb = self.input_emb(x_flat) # [B*T, N, Hidden]
        
        # Create Adjacency: Fully connected for now (Attention learns weights)
        # DenseGAT expects adj of shape [B, N, N]
        adj = torch.ones((B*T, N, N), device=x.device)
        
        # Mask out invalid nodes in adjacency (optional but good)
        # mask: [B, N]. broadcast to [B*T, N, N]
        # expanded_mask = mask.unsqueeze(1).repeat(T, 1).view(B*T, N)
        # adj = adj * expanded_mask.unsqueeze(2) * expanded_mask.unsqueeze(1)
        
        # For now, standard GATv2 is not available in Dense form in older PyG versions. 
        # We will stick to a loop or reshaped sparse GAT.
        
        # --- Simplified Sparse Approach ---
        # 1. Create one Edge Index for a "Sample Graph" of size N (fully connected)
        edge_index_template = []
        for i in range(N):
            for j in range(N):
                edge_index_template.append([i, j])
        edge_index_base = torch.tensor(edge_index_template, device=x.device).t() # [2, E]
        
        # 2. Replicate for every sample in B*T? Too big.
        # 3. Just loop T. B*N nodes is manageable.
        
        spatial_out = []
        
        # Pre-calculate Batch Edge Index for [B, N]
        # Nodes 0..N-1 belong to batch 0. N..2N-1 to batch 1.
        # Only feasible if N is constant (padding).
        
        # To save compute, let's process:
        # Reshape x -> [B*T*N, Hidden]
        # But GAT needs edges.
        
        # Let's just loop over Time steps.
        # [B, N, Hidden] input per step.
        
        batch_edge_index = []
        start_node = 0
        for b in range(B):
            # Edges for this batch index
            # fully connected i,j in 0..N
            rows, cols = edge_index_base
            batch_edge_index.append(torch.stack([rows + start_node, cols + start_node]))
            start_node += N
        
        batch_edge_index = torch.cat(batch_edge_index, dim=1) # [2, B*E]
        
        # Apply GAT
        h_t_list = []
        for t in range(T):
            xt = x[:, t, :, :].reshape(B*N, -1) # [B*N, F]
            xt = self.input_emb(xt)
            
            xt = self.gat1(xt, batch_edge_index)
            xt = F.elu(xt)
            xt = self.gat2(xt, batch_edge_index)
            xt = F.elu(xt)
            
            h_t_list.append(xt) # [B*N, Hidden]
            
        # Stack Temporal: [B*N, T, Hidden]
        h_spatial = torch.stack(h_t_list, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(h_spatial) # [B*N, T, Hidden]
        
        # Decode
        # We want to predict trajectories. 
        # Using output of LSTM (Last Hidden State? Or Sequence?)
        # Sequence-to-Sequence. We use the updated state to predict next steps?
        # Simplified: Use LSTM output at t to predict t+1, etc.
        # But we need to predict FUTURE frames (beyond input T).
        
        # For this implementation, we take the LAST output of LSTM and project it 
        # to the required number of output frames. (Vector output)
        
        last_hidden = lstm_out[:, -1, :] # [B*N, Hidden]
        
        # But we need valid predictions for specific requested output frames. 
        # The prompt is tricky: varying output lengths.
        # Let's assume we predict a fixed horizon or match training 'y'.
        # We will project last_hidden to [Output_Frames, 2] via a larger MLP 
        # or run LSTM auto-regressively. Auto-regressive is best but slow.
        # Let's use a simple multi-step dense decoder.
        
        return last_hidden

class AutoRegressiveSTGAT(STGATModel):
    def forward(self, x, mask, num_out_frames):
        # Do the spatial encoding as planned
        B, T, N, _ = x.shape
        
        # Reconstruct Batch Edge Index (Fully Connected for N nodes per batch)
        # N should include padding
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                rows.append(i); cols.append(j)
        base_edge = torch.tensor([rows, cols], device=x.device) # [2, N^2]
        
        # Offset for batches
        batch_edges = []
        for b in range(B):
            batch_edges.append(base_edge + b * N)
        full_edge_index = torch.cat(batch_edges, dim=1) # [2, B * N^2]
        
        # 1. Spatial Encoding per timestep
        spatial_states = []
        for t in range(T):
            xt = x[:, t, :, :] # [B, N, F]
            xt_flat = xt.reshape(B*N, -1)
            
            xt_emb = F.relu(self.input_emb(xt_flat))
            
            # GAT Ops
            h = self.gat1(xt_emb, full_edge_index)
            h = F.elu(h)
            h = self.gat2(h, full_edge_index)
            h = F.elu(h)
            
            spatial_states.append(h)
            
        # Stack: [B*N, T, Hidden]
        spatial_seq = torch.stack(spatial_states, dim=1)
        
        # 2. Temporal Encoding
        # Get LSTM final state
        # In: [Batch, Seq, Feat]
        _, (h_n, c_n) = self.lstm(spatial_seq) 
        # h_n: [Layers, B*N, Hidden]
        
        # 3. Decoding (Autoregressive-ish or simple Multi-MLP)
        # We need [B, T_out, N, 2]
        # Let's iterate prediction for num_out_frames
        
        preds = []
        current_state = h_n[-1] # [B*N, Hidden]
        
        # Use a Cell that updates state?
        # Simpler: One MLP maps [Hidden] -> [T_out * 2] then reshape
        # This assumes fixed T_out. But T_out varies.
        # Let's assume T_out max = 50 (5 seconds).
        
        # We'll stick to a simpler non-AR decoder for stability in training loop
        # Decoder maps [Hidden] -> [2] (Velocity/Pos)
        # But we need trajectory.
        
        # Hack: Repeat the context vector and add a time embedding?
        # Or just run an LSTM decoder.
        
        decoder_input = current_state.unsqueeze(1).repeat(1, num_out_frames, 1) # [B*N, T_out, Hidden]
        # Optionally add positional encoding here
        
        decoded, _ = self.lstm(decoder_input, (h_n, c_n)) # Continue LSTM? 
        # This is essentially "predicting from context".
        
        # Project to 2D
        # decoded: [B*N, T_out, Hidden]
        out_flat = self.decoder(decoded) # [B*N, T_out, 2]
        
        # Reshape back to [B, T_out, N, 2]
        out_reshaped = out_flat.view(B, N, num_out_frames, 2)
        return out_reshaped.permute(0, 2, 1, 3) # [B, T_out, N, 2]

# --- Train ---

def train():
    print("Initializing Data Pipeline...")
    dataset = NFLBigDataDataset(CONFIG['train_dir'], mode='train')
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate)
    
    print("Building Model...")
    sample_batch = next(iter(loader))
    feat_dim = sample_batch['x'].shape[-1]
    
    model = AutoRegressiveSTGAT(
        in_channels=feat_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    # Train Loop
    print(f"Starting Training for {CONFIG['epochs']} epochs...")
    model.train()
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            x = batch['x'].to(CONFIG['device']) # [B, T, N, F]
            y = batch['y'].to(CONFIG['device']) # [B, T_out, N, 2]
            
            optimizer.zero_grad()
            
            # Predict
            # Determine frames needed
            num_out_frames = y.shape[1]
            preds = model.forward(x, None, num_out_frames) # [B, T_out, N, 2]
            
            # Loss
            loss = criterion(preds, y)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}")
        
        # Save Metadata
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print("Training Complete & Model Saved.")

if __name__ == "__main__":
    train()
