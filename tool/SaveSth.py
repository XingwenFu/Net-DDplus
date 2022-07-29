import torch
import numpy as np
# =============================================================================
def saveModel(path, model, epoch, test_acc, train_loss, best_prec):
    best_prec = np.array(best_prec)
    try:
        precmax = np.load(path+'/best_prec.npy')
        
    except:
        precmax = 0
        precmax = np.array(precmax)
    
    if precmax < best_prec:
        precmax = best_prec
        np.save(path+'/best_prec.npy', best_prec)
        torch.save(model.state_dict(), path+'/model.pth')
        # torch.save(model.state_dict(), 'resnet.ckpt')
    print('the best Prediction accuracy is:', precmax)
    epoch_save = np.array(epoch)
    np.save(path+'/learning_rate.npy', epoch_save)
    test_acc = np.array(test_acc)
    np.save(path+'/test_acc.npy', test_acc)
    train_loss = np.array(train_loss)
    np.save(path+'/train_loss.npy', train_loss)