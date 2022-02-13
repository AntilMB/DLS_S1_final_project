import matplotlib.pyplot as plt

def show_images(image):
    plt.imshow(image['x'].squeeze().detach().cpu().permute(1, 2, 0))
    plt.show()
    plt.imshow(image['y'].squeeze().detach().cpu().permute(1, 2, 0))
    plt.show()

def show_image(image):
    plt.imshow(image.squeeze().detach().cpu().permute(1, 2, 0))
    # plt.show()


def show_epoch_res(pred):
    fig = plt.figure(figsize=(20, 12))
    
    fig.add_subplot(2, 3, 1)
    show_image(pred['real_x'])
    plt.title('real_x')
    plt.axis('off')
    
    fig.add_subplot(2, 3, 2)
    show_image(pred['fake_y'])
    plt.title('fake_y')
    plt.axis('off')
    
    fig.add_subplot(2, 3, 3)
    show_image(pred['rec_x'])
    plt.title('rec_x')
    plt.axis('off')
    
    
    fig.add_subplot(2, 3, 4)
    show_image(pred['real_y'])
    plt.title('real_y')
    plt.axis('off')
    
    fig.add_subplot(2, 3, 5)
    show_image(pred['fake_x'])
    plt.title('fake_x')
    plt.axis('off')
    
    fig.add_subplot(2, 3, 6)
    show_image(pred['rec_y'])
    plt.title('rec_x')
    plt.axis('off')