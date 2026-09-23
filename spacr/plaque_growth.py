"""Experimental plaque growth approximations with explicit provenance.

The anchor is the median, across three independent RH/HFF vehicle-control
experiments, of the median equivalent diameter of the largest quarter of
plaques. Kelsen et al. (2023), PLOS Biology 21(5): e3002110, S20 Data, CC BY.
https://doi.org/10.1371/journal.pbio.3002110

528 measured plaques support a seven-day size anchor, not a temporal growth
curve. Linear diameter growth through zero is an assumption. Leave-one-
experiment-out endpoint time error averages 39.73 hours (range 29.69–54.34).
The reproducible extraction and assessment live in tools/build_plaque_growth_reference.py.
"""
from __future__ import annotations

import math
import statistics

REFERENCE_DIAMETER_UM = 893.8178699548309
REFERENCE_HOURS = 168.0
REFERENCE_ID = 'kelsen2023_rh_hff_vehicle_day7_v1'
REFERENCE_URL = 'https://doi.org/10.1371/journal.pbio.3002110'


def largest_quarter(areas):
    """Median equivalent diameter of the largest ceil(N/4) positive areas.

    :param areas: finite positive plaque areas, all in the same squared unit.
    :returns: diameter, total count and selected count; diameter is None for
        fewer than four plaques. Invalid/nonpositive areas raise ValueError.
    """
    values = [float(a) for a in areas]
    if any(not math.isfinite(a) or a <= 0 for a in values):
        raise ValueError('Plaque areas must be finite and positive.')
    count = len(values)
    selected = math.ceil(count / 4) if count >= 4 else 0
    diameter = statistics.median(math.sqrt(4 * a / math.pi)
                                 for a in sorted(values)[-selected:]) if selected else None
    return dict(diameter=diameter, count=count, selected=selected)


def estimate_page(wells, *, reference_diameter_um=REFERENCE_DIAMETER_UM,
                  reference_hours=REFERENCE_HOURS):
    """Suggest missing scale/time using an explicitly assumed growth relation.

    :param wells: mappings with unique ``well`` identifiers, ``areas_px`` and
        optional measured ``pixels_per_um`` and ``formation_hours``. A scale
        must be positive; time may be zero. Do not pass earlier estimates as
        measured input. Different resolutions are converted before pooling.
    :param reference_diameter_um: largest-quarter reference diameter in µm.
    :param reference_hours: its positive elapsed time in hours.
    :returns: report with per-well ``estimated_*`` values, sources, counts,
        observed page descriptor and assumptions. Measured inputs are never
        overwritten. With neither scale nor time, reference duration is an
        explicitly labelled assumption; pixel scale follows from that choice.

    The whole-page descriptor assumes a common formation time. Conflicting
    known times disable page pooling; each calibrated well is then evaluated
    separately. Unknown-scale wells use their own pixel descriptor. Physical
    measurements are never computed from these suggestions automatically.
    """
    from .plaque_papers import calibration_number

    rd = calibration_number(reference_diameter_um, name='reference_diameter_um')
    rh = calibration_number(reference_hours, name='reference_hours')
    if rd is None or rh is None:
        raise ValueError('A positive reference diameter and time are required.')
    prepared = []
    identifiers = set()
    for well in wells:
        key = well['well']
        if key in identifiers:
            raise ValueError('Well identifiers must be unique.')
        identifiers.add(key)
        areas = list(well['areas_px'])
        descriptor = largest_quarter(areas)
        scale = calibration_number(well.get('pixels_per_um'), name='pixels_per_um')
        hours = calibration_number(well.get('formation_hours'), name='formation_hours', allow_zero=True)
        prepared.append(dict(well=key, areas=areas, descriptor=descriptor, scale=scale, hours=hours))
    known_times = {w['hours'] for w in prepared if w['hours'] is not None}
    pooled = [(float(area) / w['scale'] ** 2, w['well']) for w in prepared if w['scale']
              for area in w['areas']]
    page = largest_quarter([area for area, _ in pooled]) if len(known_times) <= 1 else largest_quarter([])
    selected_wells = {}
    if page['selected']:
        for _, key in sorted(pooled, key=lambda item: item[0])[-page['selected']:]:
            selected_wells[str(key)] = selected_wells.get(str(key), 0) + 1
    out = []
    for w in prepared:
        descriptor = w['descriptor']
        result = dict(well=w['well'], estimated_pixels_per_um=None, estimated_formation_hours=None,
                      estimation_source='insufficient plaques (at least four required)',
                      plaque_count=descriptor['count'], selected_count=descriptor['selected'])
        if w['scale'] is not None and w['hours'] is not None:
            result['estimation_source'] = 'measured scale and entered time retained'
        elif descriptor['diameter'] is not None:
            time = w['hours']
            source = 'entered formation time'
            if time is None:
                if len(known_times) == 1:
                    time = next(iter(known_times))
                    source = 'shared page time (assumes simultaneous formation)'
                elif page['diameter'] is not None:
                    time = rh * page['diameter'] / rd
                    source = 'page largest-quarter physical diameter; assumed linear growth'
                elif w['scale']:
                    time = rh * descriptor['diameter'] / w['scale'] / rd
                    source = 'well largest-quarter physical diameter; assumed linear growth'
                else:
                    time = rh
                    source = 'assumed reference duration; time not identifiable from pixels alone'
                result['estimated_formation_hours'] = time
            if w['scale'] is None and time > 0:
                result['estimated_pixels_per_um'] = descriptor['diameter'] / (rd * time / rh)
            result['estimation_source'] = source
        out.append(result)
    return dict(experimental=True, temporal_validation=False, reference_id=REFERENCE_ID,
                reference_url=REFERENCE_URL, reference_diameter_um=rd, reference_hours=rh,
                reference_modified=rd != REFERENCE_DIAMETER_UM or rh != REFERENCE_HOURS,
                assumption='Linear equivalent-diameter growth through zero; matched RH/HFF control biology assumed.',
                endpoint_mae_hours=39.727144313355254,
                endpoint_validation='Three held-out seven-day experiments; not validation at other times or for custom references.',
                page_diameter_um=page['diameter'], page_selected_count=page['selected'],
                page_selected_wells=selected_wells, wells=out)


def estimates_from_settings(wells, settings):
    """Return per-well suggestions and complete JSON provenance when enabled.

    :param wells: :func:`estimate_page` input mappings.
    :param settings: ``plaque_estimate_growth`` (default False), optional
        ``plaque_growth_reference_um`` and ``plaque_growth_reference_hours``.
    :returns: mapping by well identifier. Empty when estimation is disabled.
    """
    import json

    if not settings.get('plaque_estimate_growth', False):
        return {}
    report = estimate_page(wells,
        reference_diameter_um=settings.get('plaque_growth_reference_um', REFERENCE_DIAMETER_UM),
        reference_hours=settings.get('plaque_growth_reference_hours', REFERENCE_HOURS))
    provenance = json.dumps(report, sort_keys=True)
    return {row['well']: dict(estimated_pixels_per_um=row['estimated_pixels_per_um'],
                             estimated_formation_hours=row['estimated_formation_hours'],
                             estimation_source=row['estimation_source'],
                             growth_estimate_provenance=provenance)
            for row in report['wells']}
