"""Read individual Hypersim HDF5 members without extracting a scene ZIP."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import zipfile


@dataclass(frozen=True)
class ZipMember:
    archive: Path
    member: str

    def __str__(self):
        return f"{self.archive}!/{self.member}"


def zip_frames(root, suffixes):
    """Yield complete channel groups in the same scene/camera/frame order."""
    for archive_path in sorted((Path(root) / "archives").glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = {m.filename for m in members if not m.is_dir()}
            if len(names) != sum(not m.is_dir() for m in members):
                raise ValueError(f"Duplicate Hypersim ZIP members: {archive_path}")
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe Hypersim ZIP member: {member.filename}")
            for name in sorted(names):
                rgb = PurePosixPath(name)
                if not (rgb.name.startswith("frame.") and rgb.name.endswith(".color.hdf5")
                        and rgb.parent.name.startswith("scene_cam_") and rgb.parent.name.endswith("_final_hdf5")):
                    continue
                final = rgb.parent
                scene_dir = final.parent.parent if final.parent.name == "images" else final.parent
                camera = final.name.removesuffix("_final_hdf5")
                frame = rgb.name.removesuffix(".color.hdf5")
                geometry = final.parent / (camera + "_geometry_hdf5")
                paths = [str(geometry / (frame + suffix)) for suffix in suffixes]
                if all(path in names for path in paths):
                    yield scene_dir, camera, frame, [ZipMember(archive_path, n) for n in [name, *paths]]
