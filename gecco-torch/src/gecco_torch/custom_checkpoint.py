import lightning.pytorch as pl
import os

class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self,start_epoch, *args, **kwargs):
        self.start_epoch = start_epoch
        super().__init__(*args, **kwargs)

    def format_checkpoint_name(
        self, metrics, ver= None,**kwargs,
    ) -> str:
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=0)))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=5)))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.12), filename='{epoch:d}'))
            'epoch=2.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
            ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
            ... auto_insert_metric_name=False)
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-validation_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
            >>> os.path.basename(ckpt.format_checkpoint_name({}))
            'missing=0.ckpt'
            >>> ckpt = ModelCheckpoint(filename='{step}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
            'step=0.ckpt'

        """
        filename = self._format_checkpoint_name(self.filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)
        # print(filename)

        # edit epoch
        cur_epoch = int(filename.split("=")[1].split("-")[0])
        filename = filename.replace(f"epoch={cur_epoch}", f"{self.start_epoch+cur_epoch}")
        filename = "epoch="+filename

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        # print(self.dirpath)
        # print(os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name)
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name


    # def format_checkpoint_name(self, metrics, ver=None):
    #     # You can access your metrics here and format the filename as you wish
    #     # For example, include custom variables or formats
    #     filename = f"continued_epoch={self.start_epoch+self.current_epoch}-val_loss={metrics['val_loss']:.2f}"

    #     # If you have custom variables to include that aren't in metrics, add them here
    #     # For example, if you wanted to add a custom variable 'custom_var'
    #     # custom_var = "custom_value"  # Assume you get this from somewhere
    #     # filename += f"-custom_var={custom_var}"

    #     if ver is not None:
    #         filename += f"-v{ver}"

    #     return filename