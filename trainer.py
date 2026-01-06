import torch
import os
import numpy as np
from tqdm import tqdm
from metrics import find_optimal_threshold

class BinaryTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {
            'train/loss': [],
            'train/accuracy': [],
            'val/loss': [],
            'val/accuracy': [],
            'test/apcer': None,
            'test/bpcer': None,
            'test/ace': None,
            'test/accuracy': None,
            'test/threshold': None
        }


    def epoch_train(self, model, train_loader, criterion, optimizer, train_threshold=0.5):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total_samples = 0

        for imgs, labels in tqdm(train_loader, desc="train"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(imgs)
            normalized_outputs = (outputs + 1) / 2.0
            loss = criterion(normalized_outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = (normalized_outputs > train_threshold).long()
            train_correct += (predicted == labels.long()).sum().item()
            train_total_samples += imgs.size(0)
            train_loss += loss.item()

        tran_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = (train_correct / train_total_samples) * 100.0

        self.history['train/loss'].append(tran_epoch_loss)
        self.history['train/accuracy'].append(train_epoch_acc)

        return {
            'train/loss': tran_epoch_loss,
            'train/accuracy': train_epoch_acc
        }
    

    def epoch_validate(self, model, val_loader, criterion, train_threshold=0.5):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="val"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float)

                outputs = model(imgs)
                normalized_outputs = (outputs + 1) / 2.0
                loss = criterion(normalized_outputs, labels)

                predicted = (normalized_outputs > train_threshold).long()
                val_correct += (predicted == labels.long()).sum().item()
                val_total_samples += imgs.size(0)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = (val_correct / val_total_samples) * 100.0

        self.history['val/loss'].append(val_epoch_loss)
        self.history['val/accuracy'].append(val_epoch_acc)

        return {
            'val/loss': val_epoch_loss,
            'val/accuracy': val_epoch_acc
        }
    

    def epoch_test(self, model, test_loader):
        model.eval()
        test_labels = []
        test_probabilities = []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="test"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(imgs)
                normalized_outputs = (outputs + 1) / 2.0
                probabilities = normalized_outputs

                test_probabilities.append(probabilities)
                test_labels.append(labels)

        test_labels = torch.cat(test_labels).cpu().numpy()
        test_probabilities = torch.cat(test_probabilities).cpu().numpy()
        return test_labels, test_probabilities


    def train(self, model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, model_save_path, train_threshold=0.5):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('-' * 50)

            train_results = self.epoch_train(model, train_loader, criterion, optimizer, train_threshold)
            val_results = self.epoch_validate(model, val_loader, criterion, train_threshold)

            print(f"Train Loss: {train_results['train/loss']:.4f} | Train Acc: {train_results['train/accuracy']:.2f}%")
            print(f"Val   Loss: {val_results['val/loss']:.4f} | Val   Acc: {val_results['val/accuracy']:.2f}%")

            if val_results['val/loss'] < best_val_loss:
                best_val_loss = val_results['val/loss']
                best_model_state = model.state_dict().copy()
                print(f"New best model found! Saving to {model_save_path}")
                torch.save({
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None
                }, model_save_path)

            # step scheduler
            if scheduler:
                scheduler.step()

        return best_model_state
    

    def test(self, model, test_loader, result_save_path, based_on='ace'):
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

        labels, probabilities = self.epoch_test(model, test_loader)
        threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(labels, probabilities, based_on=based_on)

        self.history['test/apcer'] = apcer
        self.history['test/bpcer'] = bpcer
        self.history['test/ace'] = ace
        self.history['test/accuracy'] = accuracy
        self.history['test/threshold'] = threshold

        print(f"APCER:    {apcer:.2f}%")
        print(f"BPCER:    {bpcer:.2f}%")
        print(f"ACE:      {ace:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Optimal Threshold: {threshold:.6f}")

        torch.save(self.history, result_save_path)


    def run(self, model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, num_epochs, model_save_path, result_save_path, train_threshold=0.5, based_on='ace'):
        # Train phase
        best_model_state = self.train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, model_save_path, train_threshold)

        # Test phase
        model.load_state_dict(best_model_state)
        self.test(model, test_loader, result_save_path, based_on)
        