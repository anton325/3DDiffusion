"""
Classes defining the generation context (conditioning image and camera intrinsics matrix) and
a training example (point cloud and context). Additional utils for pretty-printing etc.
"""
from typing import NamedTuple, List, Tuple

import torch
from torch import Tensor
from enum import Enum



class Mode(Enum):

    """
    Steuere Verhalten des Models
    """

    """
    in_world_space bedeutet, dass diffusion die PC im world space gehalten wird und nur fürs projizieren der Punkte aufs bild in den 
    Camera space überführt werden müssen
    """
    in_world_space = 1

    """
    in_camera_space ist normales vanilla gecco
    """
    in_camera_space = 2
    isotropic_rgb = 3
    normal = 4

    """
    lie_rotations die rotationen werden als lie algebra (also im tangentialspace) representiert
    Das ist aber die falsche implementierung, die richtige ist 12
    """
    lie_rotations_wrong = 5

    """
    splatting loss decides whether to use the splatting loss or not
    """
    splatting_loss = 6

    """
    Train one epoch only the xyz coordinates
    """
    warmup_xyz = 7

    """
    Es werden nur die xyz koordinaten vom dataloader ausgegeben
    """
    only_xyz = 8

    """
    Es werden die spherical harmonics in RGB konvertiert im Dataloader und im splatting dann auch wieder umgerechnet
    """
    rgb = 9

    """
    Beim Conditioning werden nur die sichtbaren Punkte (mit depth gefiltert) auf das Bild
    projiziert
    """
    visibile_filter = 10

    """
    Benutze für diffusion die euler normale 3x3 rotation matrix (um den output vom netzwerk mit roma zur nächsten validen rotation matrix zu mappen) -> in kombination mit lie rotation
    """
    procrustes = 11

    """
    lie_rotations die rotationen werden als normale rotationsmatrix SO3 dargestellt. Aber das vernoisen geht anders: Wir erstellen random noise, tun so als wäre das in so3 (also tangentialspace)
    und machen daraus eine valide rotation, indem wir es mit exp in SO3 transformieren. Dann haben wir sozusagen eine "noisy rotation" und vernoisen unsere korrekte rotation damit, indem wir sie multiplizieren
    """
    rotation_matrix_mode = 12

    """
    log_L wir versuchen die covariance matrix direkt vorherzusagen. Damit das funktioniert sagen wir nicht direkt die cov matrix voraus,
    sondern L, und cov = L L.t, wobei die L eine lower triangular matrix ist, bei der die diagonale positive ist 
    """
    log_L = 13

    """
    xyz und dann noch 11 0en
    """
    fill_xyz = 14

    """
    xyz + sh
    """
    xyz_sh = 15

    """
    xyz + scaling
    """
    xyz_scaling = 16
    """
    xyz + rotation
    """
    xyz_rotation = 17
    """
    xyz + opacity
    """
    xyz_opacity = 18

    """
    xyx + scaling + rotation
    """ 
    xyz_scaling_rotation = 19

    """
    xyz_cov 
    """
    only_xyz_cov = 20

    """
    """
    normal_gt_rotations = 21

    """
    cov matrix aber im dataloader kommt die 3x3 matrix raus
    """
    cov_matrix_3x3 = 22

    """
    alles außer rotations, die werden als gt geladen. Die scalings werden canonized zu [größtes, mittleres, kleinstes]
    """
    gt_rotations_canonical = 23


    """
    es wird das Vorgehen in SO3 diffusion nachimplementiert
    """
    so3_diffusion = 24

    """
    es wird das Vorgehen in SO3 diffusion nachimplementiert, aber mit x0 als Ziel
    """
    so3_x0 = 25

    """
    gehe den Umweg über die log cholesky aufteilung, um dann die geodesic distance zu berechnen
    """
    cholesky = 26


    """

    """
    no_rotation = 27

    """

    """
    activated_scales = 28

    """
    rotationen im lie space noisen
    """
    lie_rotations = 29

    """
    zusatz zum splatting loss: Wenn das aktiviert ist, wird der splatting loss über dem context bild, also 
    das Bild, das zum Generieren genutzt wurde, ausgewertet -> das sollte in der Evaluierung den loss auf dem bild verbessern, insgesamt aber verschlechtern
    """
    ctx_splat = 30

    """
    activated_lie: scalings sind aktiviert und die rotations werden in lie algebra vernoised (und gesampled)
    """
    activated_lie = 31

    """
    Zeichne die wie bei SO3 Plotting. Das geht natürlich nur bei manchen modi, nämlich bei denen, wo die Rotationen vorliegen - also nicht 
    Cholesky, log L und so was -> geht irgendwie nicht so gut, deswegen wird nur die rotation distance gelogged
    """
    rotational_distance = 32

    """
    so wie rotational_distance machen wir das gleiche nur für das L bei Cholesky
    """
    cholesky_distance = 33

    """
    gecco_projection -> wie die punkte auf die point cloud projiziert werden, gecco ist die vanilla art und weise
    """
    gecco_projection = 34

    """
    depth_projection -> wie die aus pc2 machen (ca.) also dass depth eliminiert wird
    """
    depth_projection = 35
    
    """
    log the gradients magnitudes
    """
    log_grads = 36

    """
    benutze den dino für feature extraction
    """
    dino = 37

    """
    """
    only_splatting_loss = 38

    """
    """
    dino_triplane = 39

    """
    """
    normal_opac = 40
    
def enum_serializer(obj):
    if isinstance(obj, Enum):
        return obj.name  # or obj.value if you prefer to serialize by value
    raise TypeError(f"Type {type(obj)} not serializable")

def _raw_repr(obj) -> list[str]:
    """
    A helper for implementing __repr__. It returns a list of lines which
    describe the object. Works recursively for objects which have a _enumerate_fields.
    The reason for returning a list of lines is to enable indented printing of
    nested objects.
    """
    lines = []
    lines.append(f"{type(obj).__name__}(")

    for name, value in obj._enumerate_fields():
        if hasattr(value, "_raw_repr"):
            head, *tail, end = value._raw_repr()
            lines.append(f" {name}={head}")
            for line in tail:
                lines.append(f"  {line}")
            lines.append(f" {end}")
        elif torch.is_tensor(value):
            lines.append(f" {name}={tuple(value.shape)},")
        else:
            lines.append(f" {name}={value},")

    lines.append(f")")
    return lines


def apply_to_tensors(obj: object, f: callable) -> object:
    """
    Applies a function `f` to all tensors in the object. Works out-of-place
    """
    applied = {}
    for name, value in obj._enumerate_fields():
        if hasattr(value, "apply_to_tensors"):
            applied[name] = value.apply_to_tensors(f)
        elif torch.is_tensor(value):
            applied[name] = f(value)
        else:
            applied[name] = value

    return type(obj)(**applied) # instantiate a class of type object but with the new parameters


class DataError(RuntimeError):
    pass


def _named_tuple_enumerate_fields(obj: NamedTuple):
    yield from obj._asdict().items()

class Camera(NamedTuple):

    world_view_transform: Tensor
    projection_matrix: Tensor
    tanfovx: float
    tanfovy: float
    imsize: int

class InsInfo(NamedTuple):
    category: str
    instance: str


class Context3d(NamedTuple):
    """
    A class representing the context of a generation. It consists of a conditioning
    image and a camera matrix. The camera intrisics matrix is 3x3.
    """

    image: Tensor
    K: Tensor
    w2c: Tensor
    category: str

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())
    
class Mask(NamedTuple):
    mask: Tensor
    
class GaussianContext3d(NamedTuple):
    """
    A class representing the context of a generation. It consists of a conditioning
    image and a camera matrix. The camera intrisics matrix is 3x3.
    """

    image: Tensor
    K: Tensor
    c2w: Tensor
    w2c: Tensor
    camera: Camera
    splatting_cameras: List[Tuple[Camera, Tensor]] # liste von tuplen mit camera informationen und dem bild
    mask_points: Mask
    insinfo: InsInfo

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())




class Example(NamedTuple):
    """
    A class representing a training example. It consists of a point cloud and a context (possibly None).
    """

    data: Tensor
    ctx: Context3d | None

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())
    
    
class GaussianExample(NamedTuple):
    """
    A class representing a training example. It consists of a point cloud and a context (possibly None).
    """

    data: Tensor
    ctx: GaussianContext3d | None

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())