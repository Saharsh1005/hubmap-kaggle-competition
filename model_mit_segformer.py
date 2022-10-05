# -*- coding: utf-8 -*-
from daformer import *
from mit import *

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
	IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
	
	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x - self.mean) / self.std
		return x

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.xavier_normal_(m.weight, gain=1)
        #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        

class Net(nn.Module):
    
	def __init__(self,
				 encoder=mit_b2,
				 decoder=daformer_conv1x1, 
				 encoder_cfg={},
				 decoder_cfg={},
	             ):  
		super(Net, self).__init__()
        
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		self.output_type = ['loss','inference']
		self.rgb = RGB()
		self.encoder = encoder() # drop_path_rate =0.3,
		self.dropout = nn.Dropout(p = 0.1) 
		encoder_dim = self.encoder.embed_dims

		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, 1, kernel_size=1),
		)

		self.aux = nn.ModuleList([
            nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(4)
        ])
        
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		# self.cls_head = nn.Linear(320,5,bias = False)
		self.cls_head = nn.Sequential(
							nn.BatchNorm1d(320).apply(init_weight),
							nn.Linear(320, 128 ).apply(init_weight),
							
							nn.ReLU(inplace=True),
							nn.BatchNorm1d(128).apply(init_weight),
							nn.Linear(128, 5).apply(init_weight)
						)
		self.is_train = True


	def forward(self, batch):
		
		image = batch['image'] 
		image = self.rgb(image)
		
		B,C,H,W = image.shape
		encoder = self.encoder(image)
		#print([f.shape for f in encoder])
		
		last, decoder = self.decoder(encoder)
        
		if self.is_train is True:
		    # print('Dropout happened! ')
		    last  = self.dropout(last) 
            
		logit = self.logit(last)
		logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		
		output = {}
		if 'loss' in self.output_type:
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
			for i in range(4):
				output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](encoder[i]),batch['mask'])
		 
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit)
		
		return output


def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss
 
def run_check_net():
	batch_size = 2
	image_size = 768
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.to(device) for k, v in batch.items()} #v.cu
	
	net = Net().to(device) #.cuda()
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
    run_check_net()