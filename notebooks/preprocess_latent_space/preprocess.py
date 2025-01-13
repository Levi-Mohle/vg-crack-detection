class HDF5CreatePatchesDataset(Dataset):
    """ A class which creates mini-patches as a Dataset object 
        from height and rgb images directly from a Keyence h5 file

    """
    def __init__(self, input_filename, idx_mini_range, patch_size
                 , inpaint=False, transform=None, down_scale_factor = 1):
        """
        Args:
            input_filename (str): name of the input file including full path and extension
            mini_idx_range ([int]) : list of integers indices of mini-patch indices 
                                    from original painting
            patch_size ([int,int]) : The new grid you want to divide the original patches in
            inpaint (bool): Pre-processing height images by inpainting 0 values
            down_scale_factor (int) : factor by which you downscale original painting patches
        """

        self.input_filename = input_filename
        self.idx_mini_range = idx_mini_range
        self.patch_size = patch_size
        self.transform = transform
        self.down_scale_factor = down_scale_factor

        # Open the input Keyence data file for reading, and read some required parameters from it
        self.kr = keyence.Read(self.input_filename, verbose=False)

        # Convert mini-patch indices to main patch indices
        indices = index_libary(self.kr.num_images(), self.patch_size)
        self.idx_mini_range.sort()
        self.idx_range = []
        self.mini_patch_indices = []
        for idx in self.idx_mini_range:
            main, row, col = np.where(indices == idx)
            self.idx_range.append(main[0])
            self.mini_patch_indices.append([row[0], col[0]])

        # Prevent opening same main patch twice
        self.idx_range_unique = list(set(self.idx_range))
        self.idx_range_unique.sort()

        self.num_rows                = int(self.kr.num_rows()/self.down_scale_factor)
        self.num_cols                = int(self.kr.num_cols()/self.down_scale_factor)

        if self.num_rows // self.patch_size[0] == 0:
            log.error(f"Original dimension of {self.num_rows} not divisible by {self.patch_size[0]}")
        elif self.num_cols // self.patch_size[1] == 0:
            log.error(f"Original dimension of {self.num_cols} not divisible by {self.patch_size[1]}")

        self.chunk_size = [self.num_rows // self.patch_size[0],
                           self.num_cols // self.patch_size[1]]

        self.total_chunks = len(self.idx_mini_range)

        # Create empty lists to fill from h5 file
        self.height = []
        self.rgb = []
        # Loop across all image patches.
        for idx_r in self.idx_range_unique:
            log.info(f"Reading image and position data for index {idx_r}")

            height_data = cv2.resize(self.kr.height_image(idx_r),
                                        (self.num_rows, self.num_cols),
                                        interpolation=cv2.INTER_NEAREST)

            # Fill in 0 values by inpaint
            if inpaint:
                # Calculate the lowerbound of the 99.7% confidence interval
                int99 = np.mean(height_data) - 3 * np.std(height_data)
                mask = (height_data <= int99).astype('uint8')
                height_data = cv2.inpaint(height_data, mask, 10, cv2.INPAINT_NS)

            rgb_data = cv2.resize(self.kr.rgb_image(idx_r),
                                    (self.num_rows, self.num_cols),
                                    interpolation=cv2.INTER_CUBIC)
            # position_data = self.kr.position_m(idx_r)
            self.height.append(height_data)
            self.rgb.append(rgb_data)

        # Concatenate all rgb and height images separate
        self.height = np.expand_dims(np.stack(self.height), axis=1) # size [num_rows, num_cols, 1]
        self.rgb = np.stack(self.rgb).transpose(0,3,1,2) # size [num_rows, num_cols, 3]

        # Concatenate rgb and height images together
        # self.all = np.concatenate((self.rgb, np.expand_dims(self.height, axis=-1)), axis=-1) # size [num_rows, num_cols, 4]

        # Close the Keyence file for reading
        self.kr.close()

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx: int) -> tuple[any, any]:

        # Calculate which image and which chunk within the image
        img_idx = assign_indices(self.idx_range)[idx]
        row_idx = self.mini_patch_indices[idx][0]
        col_idx = self.mini_patch_indices[idx][1]

        row_start = row_idx * self.chunk_size[0]
        col_start = col_idx * self.chunk_size[1]

        height_chunks = self.height[img_idx, :, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1]]
        rgb_chunks = self.rgb[img_idx, :, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1]]

        # all_chunks = self.all[img_idx, row_start:row_start + self.chunk_size[0], col_start: col_start+self.chunk_size[1], :]
        minipatch_nr = self.idx_mini_range[idx]

        if self.transform is not None:
            all_chunks = self.transform(all_chunks)

        return rgb_chunks, height_chunks, minipatch_nr # all_chunks