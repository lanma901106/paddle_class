import paddle
import paddle.nn as nn
from resnet18 import ResNet18
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter

# target: 93%+


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=20):
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)

        if batch_idx > 0 and batch_idx % report_freq == 0:
            print(f'----- Batch[{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')

    print(f'----- Epoch[{epoch}/{total_epoch}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')


def validate(model, dataloader, criterion, report_freq=10):
    print('----- Validation')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        pred = paddle.nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))
        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)

        if batch_idx > 0 and batch_idx % report_freq == 0:
            print(f'----- Batch [{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')

    print(f'----- Validation Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')


def main():
    total_epoch = 200
    batch_size = 512
    # batch_size = 256

    model = ResNet18()
    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size, mode='train')
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(val_dataset, batch_size, mode='test')
    criterion = nn.CrossEntropyLoss()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)
    # scheduler = segmentation.optimizer.lr.CosineAnnealingDecay(0.01, total_epoch)
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                          parameters=model.parameters(),
                                          momentum=0.9,
                                          weight_decay=5e-4)

    eval_mode = False
    if eval_mode:
        state_dict = paddle.load('./resnet18_ep200.pdparams')
        model.set_state_dict(state_dict)
        validate(model, val_dataloader, criterion)
        return

    save_freq = 50
    test_freq = 10
    for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, total_epoch)
        scheduler.step()

        if epoch % test_freq == 0 or epoch == total_epoch:
            validate(model, val_dataloader, criterion)

        if epoch % save_freq == 0 or epoch == total_epoch:
            paddle.save(model.state_dict(), f'./resnet18_ep{epoch}.pdparams')
            paddle.save(optimizer.state_dict(), f'./resnet18_ep{epoch}.pdopts')


if __name__ == "__main__":
    main()
