
# data_root = '/dataset/vfayezzhang/dataset/LED/train/'
# ll_dir = os.path.join(data_root, 'input')
# rgb_dir = os.path.join(data_root, 'target')
# depth_dir = os.path.join(data_root, 'depth')
# ll_p_depth_dir = os.path.join(data_root, 'll_p_depth')
# he_ll_p_depth_dir = os.path.join(data_root, 'he_ll_p_depth')
# p_depth_dir = os.path.join(data_root, 'p_depth')
#
# files = [f for f in os.listdir(ll_p_depth_dir) if f.endswith('.jpg') or f.endswith('.png')]
#
# # 遍历每个文件
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
#
# for file in files[:10]:
#     # Construct file paths
#     ll_path = os.path.join(ll_dir, file)
#     rgb_path = os.path.join(rgb_dir, file)
#     depth_path = os.path.join(depth_dir, file)
#     ll_p_depth_path = os.path.join(ll_p_depth_dir, file)
#     he_ll_p_depth_path = os.path.join(he_ll_p_depth_dir, file)
#     p_depth_path = os.path.join(p_depth_dir, file)
#
#     # Ensure all files exist
#     if all(os.path.exists(path) for path in
#            [ll_path, rgb_path, depth_path, ll_p_depth_path, he_ll_p_depth_path, p_depth_path]):
#         # Open images
#         ll_img = Image.open(ll_path)
#         rgb_img = Image.open(rgb_path)
#         depth_img = Image.open(depth_path)
#         ll_p_depth_img = Image.open(ll_p_depth_path).convert('L')
#         he_ll_p_depth_img = Image.open(he_ll_p_depth_path).convert('L')
#         p_depth_img = Image.open(p_depth_path).convert('L')
#
#         # Create a figure to display subplots
#         fig, axs = plt.subplots(2, 3, figsize=(30, 24))
#         fig.patch.set_facecolor('white')  # Set background color
#
#         # Set images and titles with larger font size
#         axs[0, 0].imshow(ll_img)
#         axs[0, 0].set_title('Low Light', fontsize=20)  # Increase font size
#
#         axs[0, 1].imshow(rgb_img)
#         axs[0, 1].set_title('Normal Light', fontsize=20)  # Increase font size
#
#         axs[0, 2].imshow(depth_img, cmap='gray')
#         axs[0, 2].set_title('GT Depth', fontsize=20)  # Increase font size
#
#         axs[1, 0].imshow(ll_p_depth_img, cmap='gray')
#         axs[1, 0].set_title('Low Light Depth', fontsize=20)  # Increase font size
#
#         axs[1, 1].imshow(he_ll_p_depth_img, cmap='gray')
#         axs[1, 1].set_title('LL Depth with HE', fontsize=20)  # Increase font size
#
#         axs[1, 2].imshow(p_depth_img, cmap='gray')
#         axs[1, 2].set_title('Normal Light Depth', fontsize=20)  # Increase font size
#
#         # Turn off axes and add a blue border
#         for ax in axs.flat:
#             ax.axis('off')
#             for spine in ax.spines.values():
#                 spine.set_edgecolor('blue')  # Set border color to blue
#                 spine.set_linewidth(10)  # Set border width to be thicker
#
#         # Adjust layout and spacing
#         plt.tight_layout(pad=5)  # Increase spacing between subplots
#         plt.subplots_adjust(top=0.9)  # Adjust top spacing if necessary
#
#         # Save the concatenated image
#         output_path = os.path.join(data_root, 'output', f"{os.path.splitext(file)[0]}.jpg")
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path)
#         plt.close()
#
# print("处理完成！")
