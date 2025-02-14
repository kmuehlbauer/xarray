from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from cf_codecs import conventions

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendEntrypoint,
)
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    import os

    from xarray.core.types import ReadBuffer


class StoreBackendEntrypoint(BackendEntrypoint):
    description = "Open AbstractDataStore instances in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html"

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
    ) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        assert isinstance(filename_or_obj, AbstractDataStore)

        # SOCF: the following lines can be split out
        # either as partial func which just gets called here
        # or as tuple(func, kwargs) which get's called here
        decode_cf = dict(
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )
        from functools import partial

        if any(decode_cf.values()):
            func = partial(conventions.decode_cf, **decode_cf)
        else:
            # this is just the plain code which just does no decoding at all
            def no_decode(filename_or_obj, drop_variables=None):
                vars, attrs = filename_or_obj.load()
                encoding = filename_or_obj.get_encoding()
                coord_names = set()
                if isinstance(drop_variables, str):
                    drop_variables = [drop_variables]
                elif drop_variables is None:
                    drop_variables = []
                drop_variables = set(drop_variables)

                new_vars = {}
                for k, v in vars.items():
                    if k in drop_variables:
                        continue
                    new_vars[k] = v

                ds = Dataset(new_vars, attrs=attrs)
                ds = ds.set_coords(coord_names.intersection(vars))
                ds.set_close(filename_or_obj.close)
                ds.encoding = encoding

                return ds

            func = partial(no_decode)

        ds = func(filename_or_obj, drop_variables=drop_variables)

        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
