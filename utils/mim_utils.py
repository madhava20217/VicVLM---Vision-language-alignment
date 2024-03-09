import torch

# utils
def get_patches(batch, img_dim, num_patches, size_patch):
    '''Function to get patches from an image
    
    Arguments:
    1. batch: batch of tensor images (N, C, H, W)
    2. num_patches: number of patches per side
    3. size_patch: number of pixels along each side of the patch
    
    Returns:
    Tensor containing patch-wise representation of each image in the batch
    '''
    unfold_dim = int(img_dim/size_patch)
    fold_params = {'kernel_size' : unfold_dim,
                'dilation': size_patch}
    return torch.nn.functional.unfold(
        batch,
        **fold_params
        ).reshape(batch.size(0), -1, num_patches**2, size_patch**2).permute(0,2,1,3)

def get_img_from_patches(batch, img_dim, size_patch):
    '''Function to reconstruct an image from its patches
    
    Arguments:
    1. batch: batch of tensor images (N, C, H, W)
    2. size_patch: number of pixels along each side of the patch
    
    Returns:
    tensor containing reconstructed images'''
    unfold_dim = int(img_dim/size_patch)
    fold_params = {'kernel_size' : unfold_dim,
                'dilation': size_patch}
    return torch.nn.functional.fold(
        batch.permute(0,2,1,3).reshape(batch.size(0), -1, size_patch**2),
        output_size = (img_dim, img_dim),
        **fold_params
    )
    
def create_masked_image(img, mask):
    mask = (mask .repeat_interleave(16, 1)
                .repeat_interleave(16, 2)
                .unsqueeze(1)
                .contiguous()
            )
    return (img * (1 - mask))