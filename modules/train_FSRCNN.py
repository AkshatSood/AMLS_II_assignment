import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from modules.FSRCNN import FSRCNN
from modules.dataset import FSRCNNTrainData, FSRCNNValidationDataset
from helpers.helpers import AverageMeter, Helpers
from helpers.utility import Utility

from constants import TRAIN_FSRCNN_TARGETS

class FSRCNNTrainer:

    def __init__(self):
        self.helper = Helpers()
        self.utility = Utility()

    def __train_model(self, train_dataset_file, validation_dataset_file, output_dir, scale, num_workers, prefix, learning_rate=1e-3, batch_size=16, num_epochs=20, seed=42):
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(seed)

        model = FSRCNN(scale_factor=scale).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': learning_rate * 0.1}
        ], lr=learning_rate)

        print('\t\tLoading datasets...')

        train_dataset = FSRCNNTrainData(train_dataset_file)
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True)
        validation_dataset = FSRCNNValidationDataset(validation_dataset_file)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1)

        print('\t\tLoaded datasets.')

        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_psnr = 0.0

        for epoch in range(num_epochs):
            print(f'\t\t\t[{epoch}/{num_epochs}] Starting epoch training...')
            model.train()
            epoch_losses = AverageMeter()

            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f'loss={epoch_losses.avg}')

            torch.save(model.state_dict(), os.path.join(output_dir, f'{prefix}_epoch_{epoch}.pth'))

            model.eval()
            epoch_psnr = AverageMeter()

            for data in validation_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                epoch_psnr.update(self.helper.calc_psnr(preds, labels), len(inputs))

            print('\t\t\tEval PSNR: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

        best_output = os.path.join(output_dir, f'{prefix}_best.pth')
        print('\t\tBest Epoch: {}, PSNR: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, best_output)
        print(f'\t\tWeights are saved at {best_output}')

    def train_models(self):
        print('\n>=> Traing FSRCNN models...')

        for target in TRAIN_FSRCNN_TARGETS:
            print(f'\tTraining {target["name"]} FSRCNN model...')

            self.utility.check_and_create_dir(target['weights_dir'])

            self.__train_model(
                train_dataset_file=target['train_h5'],
                validation_dataset_file=target['valid_h5'],
                output_dir=target['weights_dir'],
                scale=target['scale'],
                prefix=target['prefix'],
                num_epochs=20, 
                num_workers=4,
                batch_size=16
            )