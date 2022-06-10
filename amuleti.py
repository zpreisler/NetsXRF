import XRDXRFutils as xutils
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import tifffile
import cv2
from Amulet import Amulet

if __name__ == '__main__':
    si = np.array(tifffile.imread('F:/andrea/AIdataXRF/amuleti/outputs/Febbraio2022NuovaCassettaAmuleti1_A/Si-K.tiff'))
    plt.figure('silicio')
    plt.imshow(si)

    norm = si*255/np.max(si)
    norm = norm.astype(np.uint8)

    original = norm.copy()

    thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    print(len(list(cnts)))
    amulets_pos = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        # filter too little contours
        if w<5 or h<5: continue
        else: amulets_pos.append([x,y,w,h])
        
        cv2.rectangle(norm, (x, y), (x + w, y + h), (255,255,255), 1)

    plt.figure()
    plt.imshow(norm, cmap='pink')

    # cv2.imshow('image', norm)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    # cv2.waitKey()

    data2evaluate = xutils.DataXRF().load_h5('F:/andrea/AIdataXRF/amuleti/0-24kev/Febbraio2022NuovaCassettaAmuleti1_A.h5').data

    labels_amulets = []
    label_names = []

    lab = ['Pb-L', 'Fe-K', 'Mn-K', 'Cu-K', 'Co-K', 'Ca-K', 'Ti-K', 'S-K', 'P-K', 'Au-L', 'Al-K', 'Si-K', 'Cl-K', 'Sb-L', 'Sn-L', 'Sr-K', 'V-K']

    for l in lab:
        label_names.append(l)
        if l == 'P-K' or l == 'Al-K' or l == 'Au-L':
            labels_amulets += [np.zeros((norm.shape[0], norm.shape[1]))]
            continue
        labels_amulets += [np.array(tifffile.imread('F:/andrea/AIdataXRF/amuleti/0-24kev/NuovaCassettaAmuleti1_A_'+l+'.tif'))]

    shape = labels_amulets[0].shape
    print(np.stack(labels_amulets).shape, np.max(labels_amulets))
    labels_amulets = np.swapaxes(np.stack(labels_amulets), 0, 1)
    labels = np.swapaxes(np.stack(labels_amulets), 1, 2)
    print(labels.shape, np.max(labels))

    # for each amulet, compute total counts for all elements.
    d = data2evaluate
    l= labels
    amulets = []

    plt.figure('sum_spectrum')
    plt.title('sum_spectrum')
    plt.figure('total labels')
    plt.title('total labels')
    for s, [coord_y, coord_x, w, h] in enumerate(amulets_pos):
        # if s>2:break
        
        print('amulets_pos', coord_x, coord_y, w, h)
        amuleto = Amulet(coord_x, coord_y, w, h)
        amuleto.compute_labels(data=d, labels=l)

        amulets.append(amuleto)

        plt.figure('sum_spectrum')
        plt.plot(amuleto.sum_spectrum)
        plt.figure('total labels')
        plt.plot(lab, amuleto.total_labels)

    print(len(amulets))
    plt.show()











""" 
# load the model
base = './run/synth_amuleti_140-160micron_CNN2_ch_8_run_0/'
model_path = glob(base+'*.pth')[-1]

with open(base+'config.yaml','r') as file:
    config = yaml.load(file,Loader=yaml.FullLoader)
model = nets.CNN2(channels=config['channels'], n_outputs=len(config['labels']))

checkpoint = torch.load(model_path)
print(checkpoint['model_state_dict']['fc.0.weight'].shape, model.fc)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

for dsss in glob('F:/andrea/AIdataXRF/amuleti/0-24kev/Febbraio2022*.h5'):
    data2evaluate = xutils.DataXRF().load_h5(dsss)

    data2evaluate.metadata['labels'] = ['']
    print(data2evaluate.data.shape)
    data2evaluate.labels = np.zeros((data2evaluate.shape[0],data2evaluate.shape[1], len(config['labels'])))
    data2evaluate = data2evaluate.select_labels(config['labels'])
    print('selected_labels:', data2evaluate.metadata['labels'])

    # print(data2evaluate.labels.shape)
    data = torch.from_numpy(data2evaluate.data).reshape(-1,1,data2evaluate.data.shape[-1]).float()
    labels = torch.from_numpy(data2evaluate.labels).reshape(-1,data2evaluate.labels.shape[-1]).float()

    ds = dataset.SpectraDataset(data=data, labels=labels)

    evaluate = DataLoader(ds,
                    batch_size = config['batch_size'],
                    shuffle = False,
                    drop_last = False)
    evaluate.shape = data2evaluate.labels.shape

    image = []
    device = torch.device(config['device'])
    model.to(device)
    for i,batch in enumerate(evaluate):

        x,y = batch

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        image += [outputs.cpu().detach().numpy()]

    image = np.vstack(image)
    print('\n',image.shape)
    image = image.reshape(evaluate.shape)


    labels = data2evaluate.labels
    image = image.astype(float)
    cmap = 'jet'
    print(data2evaluate.metadata['labels'])

    outfilename = data2evaluate.metadata['path'].replace('\\', '/').split('/')[-3]
    print(outfilename)

    for i, element in enumerate(data2evaluate.metadata['labels']):
        plt.figure()
        # plt.title('model '+ element)


        os.makedirs('F:/andrea/AIdataXRF/amuleti/outputs/'+outfilename, exist_ok=True)
        os.makedirs('F:/andrea/AIdataXRF/amuleti/outputs/plt-'+outfilename, exist_ok=True)
        
        tifffile.imwrite('F:/andrea/AIdataXRF/amuleti/outputs/'+outfilename+'/'+ element+'.tiff', image[:,:,i])
        plt.imsave('F:/andrea/AIdataXRF/amuleti/outputs/plt-'+outfilename+'/'+ element+'.tiff', image[:,:,i], cmap='jet')
        # plt.imshow(Image.open('F:/andrea/AIdataXRF/amuleti/outputs/'+outfilename+'/'+ element+'.tiff'))
        # plt.show()
    
    #     plt.imshow(tifffile.imread('F:/andrea/AIdataXRF/amuleti/outputs/'+outfilename+'_'+ element+'.tiff'), cmap=cmap)

    # plt.show() """