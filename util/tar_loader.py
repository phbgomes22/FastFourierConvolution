#
# Author: jotaf98
# Link: https://github.com/jotaf98/simple-tar-dataset
#

import tarfile
from io import BytesIO
from PIL import Image, ImageFile

from torch.utils.data import Dataset, get_worker_info

try:  # make torchvision optional
  from torchvision.transforms.functional import to_tensor
except:
  to_tensor = None

try:  # make torchvision optional
  from torchvision.transforms.functional import to_tensor
except:
  to_tensor = None


ImageFile.LOAD_TRUNCATED_IMAGES = True

## ADDED for multiprocess
from multiprocessing import Pool


class UnexpectedEOFTarFile(tarfile.TarFile):
  def _load(self):
    """Read through the entire archive file and look for readable
       members.
    """
    try:
      while True:
        tarinfo = self.next()
        if tarinfo is None:
          break
    except tarfile.ReadError as e:
      assert e.args[0] == "unexpected end of data"
    self._loaded = True


class TarDataset(Dataset):
  """Dataset that supports Tar archives (uncompressed).
  Args:
    archive (string or TarDataset): Path to the Tar file containing the dataset.
      Alternatively, pass in a TarDataset object to reuse its cached information;
      this is useful for loading different subsets within the same archive.
    extensions (tuple): Extensions (strings starting with a dot), only files
      with these extensions will be iterated. Default: png/jpg/jpeg.
    is_valid_file (callable): Optional function that takes file information as
      input (tarfile.TarInfo) and outputs True for files that need to be
      iterated; overrides extensions argument.
      Example: lambda m: m.isfile() and m.name.endswith('.png')
    transform (callable): Function applied to each image by __getitem__ (see
      torchvision.transforms). Default: ToTensor (convert PIL image to tensor).
    ignore_unexpected_eof (bool): ignore Unexpected EOF when iterating the tar file
      allows working with Datasets cut to a smaller size with dd
  Attributes:
    members_by_name (dict): Members (files and folders) found in the Tar archive,
      with their names as keys and their tarfile.TarInfo structures as values.
    samples (list): Items to iterate (can be ignored by overriding __getitem__
      and __len__).
  Author: Joao F. Henriques
  """
  def __init__(self, archive, transform=to_tensor, labeled:bool=True, extensions=('.png', '.jpg', '.jpeg'),
    is_valid_file=None, ignore_unexpected_eof=False):
    if not isinstance(archive, TarDataset):
      # open tar file. in a multiprocessing setting (e.g. DataLoader workers), we
      # have to open one file handle per worker (stored as the tar_obj dict), since
      # when the multiprocessing method is 'fork', the workers share this TarDataset.
      # we want one file handle per worker because TarFile is not thread-safe.
      worker = get_worker_info()
      worker = worker.id if worker else None
      self.tar_obj = {worker: tarfile.open(archive) if ignore_unexpected_eof is False else UnexpectedEOFTarFile.open(archive)}
      self.archive = archive
      self.labeled = labeled

      # store headers of all files and folders by name
      members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
      self.members_by_name = {m.name: m for m in members}
    else:
      # passed a TarDataset into the constructor, reuse the same tar contents.
      # no need to copy explicitly since this dict will not be modified again.
      self.members_by_name = archive.members_by_name
      self.archive = archive.archive  # the original path to the Tar file
      self.tar_obj = {}  # will get filled by get_file on first access

    # also store references to the iterated samples (a subset of the above)
    self.filter_samples(is_valid_file, extensions)
    
    self.transform = transform


  def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
    """Filter the Tar archive's files/folders to obtain the list of samples.
    
    Args:
      extensions (tuple): Extensions (strings starting with a dot), only files
        with these extensions will be iterated. Default: png/jpg/jpeg.
      is_valid_file (callable): Optional function that takes file information as
        input (tarfile.TarInfo) and outputs True for files that need to be
        iterated; overrides extensions argument.
        Example: lambda m: m.isfile() and m.name.endswith('.png')
    """
    # by default, filter files by extension
    if is_valid_file is None:
      def is_valid_file(m):
        return (m.isfile() and m.name.lower().endswith(extensions))

    # filter the files to create the samples list
    self.samples = [m.name for m in self.members_by_name.values() if is_valid_file(m)]


  def __getitem__(self, index):
    """Return a single sample.
    
    Should be overriden by a subclass to support custom data other than images (e.g.
    class labels). The methods get_image/get_file can be used to read from the Tar
    archive, and a dict of files/folders is held in the property members_by_name.
    By default, this simply applies the given transforms or converts the image to
    a tensor if none are specified.
    Args:
      index (int): Index of item.
    
    Returns:
      Tensor: The image.
    """
    image = self.get_image(self.samples[index], pil=True)
    image = image.convert('RGB')  # if it's grayscale, convert to RGB
    if self.transform:  # apply any custom transforms
      image = self.transform(image)

    dummy_label = 0
    print(image.shape)
    return image, dummy_label if self.labeled else image


  def __len__(self):
    """Return the length of the dataset (length of self.samples)
    Returns:
      int: Number of samples.
    """
    return len(self.samples)


  def get_image(self, name, pil=False):
    """Read an image from the Tar archive, returned as a PIL image or PyTorch tensor.
    Args:
      name (str): File name to retrieve.
      pil (bool): If true, a PIL image is returned (default is a PyTorch tensor).
    Returns:
      Image or Tensor: The image, possibly in PIL format.
    """
    print(name)
    image = Image.open(BytesIO(self.get_file(name).read()))
    if pil:
      return image
    return to_tensor(image)

  ## PG
  def get_image_pil(self, name):
    ## PG
    return Image.open(BytesIO(self.get_file(name).read()))



  def get_text_file(self, name, encoding='utf-8'):
    """Read a text file from the Tar archive, returned as a string.
    Args:
      name (str): File name to retrieve.
      encoding (str): Encoding of file, default is utf-8.
    Returns:
      str: Content of text file.
    """
    return self.get_file(name).read().decode(encoding)


  def get_file(self, name):
    """Read an arbitrary file from the Tar archive.
    Args:
      name (str): File name to retrieve.
    Returns:
      io.BufferedReader: Object used to read the file's content.
    """
    # ensure a unique file handle per worker, in multiprocessing settings
    worker = get_worker_info()
    worker = worker.id if worker else None

    if worker not in self.tar_obj:
      self.tar_obj[worker] = tarfile.open(self.archive)

    return self.tar_obj[worker].extractfile(self.members_by_name[name])


  def __del__(self):
    """Close the TarFile file handles on exit."""
    for o in self.tar_obj.values():
      o.close()


  def __getstate__(self):
    """Serialize without the TarFile references, for multiprocessing compatibility."""
    state = dict(self.__dict__)
    state['tar_obj'] = {}
    return state



class TarImageFolder(TarDataset):
  """Dataset that supports Tar archives (uncompressed), with a folder per class.
  Similarly to torchvision.datasets.ImageFolder, assumes that the images inside
  the Tar archive are arranged in this way by default:
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png
    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png
  
  Args:
    archive (string or TarDataset): Path to the Tar file containing the dataset.
      Alternatively, pass in a TarDataset object to reuse its cached information;
      this is useful for loading different subsets within the same archive.
    root_in_archive (string): Root folder within the archive, directly below
      the folders with class names.
    extensions (tuple): Extensions (strings starting with a dot), only files
      with these extensions will be iterated. Default: png/jpg/jpeg.
    is_valid_file (callable): Optional function that takes file information as
      input (tarfile.TarInfo) and outputs True for files that need to be
      iterated; overrides extensions argument.
      Example: lambda m: m.isfile() and m.name.endswith('.png')
    transform (callable): Function applied to each image by __getitem__ (see
      torchvision.transforms). Default: ToTensor (convert PIL image to tensor).
  Attributes:
    samples (list): Image file names to iterate.
    targets (list): Numeric label corresponding to each image.
    class_to_idx (dict): Maps class names to numeric labels.
    idx_to_class (dict): Maps numeric labels to class names.
    members_by_name (dict): Members (files and folders) found in the Tar archive,
      with their names as keys and their tarfile.TarInfo structures as values.
  Author: Joao F. Henriques
  """
  def __init__(self, archive, transform=to_tensor, extensions=('.png', '.jpg', '.jpeg'),
    is_valid_file=None, root_in_archive=''):
    # ensure the root path ends with a slash
    if root_in_archive and not root_in_archive.endswith('/'):
      root_in_archive = root_in_archive + '/'
    self.root_in_archive = root_in_archive

    # load the archive meta information, and filter the samples
    super().__init__(archive=archive, transform=transform, is_valid_file=is_valid_file)

    # assign a label to each image, based on its top-level folder name
    self.class_to_idx = {}
    self.targets = []
    for filename in self.samples:
      # extract the class name from the file's path inside the Tar archive
      if self.root_in_archive:
        assert filename.startswith(root_in_archive)  # sanity check (filter_samples should ensure this)
        filename = filename[len(root_in_archive):]  # make path relative to root
      (class_name, _, _) = filename.partition('/')  # first folder level

      # assign increasing label indexes to each class name
      label = self.class_to_idx.setdefault(class_name, len(self.class_to_idx))
      self.targets.append(label)
    
    if len(self.class_to_idx) == 0:
      raise IOError("No classes (top-level folders) were found with the given criteria. The given\n"
        "extensions, is_valid_file or root_in_archive are too strict, or the archive is empty.")

    # elif len(self.class_to_idx) == 1:
    #   raise IOError(f"Only one class (top-level folder) was found: {next(iter(self.class_to_idx))}.\n"
    #     f"To choose the correct path in the archive where the label folders are located, specify\n"
    #     f"root_in_archive in the TarImageFolder's constructor.")
    
    # the inverse mapping is often useful
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    ## Get all images - PG
   # self.load_all_images()


  def load_all_images(self):
    ## - Trying multiprocess
    with Pool(8) as pool:
      self.images = pool.map(self.get_image_pil, self.samples)

    print("Tar Images loaded!")


  def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
    """In addition to TarDataset's filtering by extension (or user-supplied),
    filter further to select only samples within the given root path."""
    super().filter_samples(is_valid_file, extensions)
    self.samples = [filename for filename in self.samples if filename.startswith(self.root_in_archive)]
    ## - pg (28/12/22) - fix code for celeba
    self.samples = [name.replace("._", "") for name in self.samples]

  def __getitem__(self, index):
    """Return a single sample.
    By default, this simply applies the given transforms or converts the image to
    a tensor if none are specified.
    Args:
      index (int): Index of item.
    
    Returns:
      tuple[Tensor, int]: The image and the corresponding label index.
    """
    image = self.get_image(self.samples[index], pil=True) ## - ENHANCE - using cpu everytime
    image = image.convert('RGB')  # if it's grayscale, convert to RGB
    if self.transform:  # apply any custom transforms
      image = self.transform(image)
    
    label = self.targets[index]

    return (image, label)