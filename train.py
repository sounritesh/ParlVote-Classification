from .utils import ParlDataset, Engine
import numpy as np
from torch.utils.data import DataLoader
import torch
import optuna

# import model of choice
from .lstm_mlp import LSTM_MLP

folder_path = 'gdrive/MyDrive/parl_vote_dataset/'
feature_path = folder_path+'meaned_LSTM_sentBertFeatures.npy'
label_path = folder_path+'meaned_LSTM_sentBertLabels.npy'

X = np.load(feature_path)
y = np.load(label_path)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100

def objective(trial):
    params = {
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 18, 768),
        'hidden_size': trial.suggest_int('hidden_size', 18, 768),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),

    }
    return run_training(params, False)

def run_training(params, save_model=False):
    train_ds = ParlDataset(X[0:-5500], y[0:-5500])
    val_ds = ParlDataset(X[-5500:-2000], y[-5500:-2000])

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=True)

    #initialize model of choice
    model = LSTM_MLP(  
                    lstm_layers=params['lstm_layers'], 
                    hidden_size=params['hidden_size'],
                    lstm_hidden_size=params['lstm_hidden_size'],
                    dropout=params['dropout'],
                    num_layers=params['num_layers']
                  )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    eng = Engine(model, optimizer, device=DEVICE)

    best_loss = np.inf

    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_dl)
        valid_loss = eng.evaluate(val_dl)

        print(f"Epoch: {epoch}, Train Acc: {train_loss}, Valid Acc: {valid_loss}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), folder_path+"best_lstm_model.bin")
        
        else:
            early_stopping_counter += 1

        if early_stopping_iter < early_stopping_counter:
            break

    return best_loss

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    trial_ = study.best_trial

    print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

    score = run_training(trial_.params, True)

    print(score)

if __name__=='__main__':
    main()