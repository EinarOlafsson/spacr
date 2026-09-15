{% set helpers = spacr_nested_helpers.get(obj.id, []) %}
{% if helpers %}
Nested helpers
--------------

{% for helper in helpers %}
.. py:function:: {{ helper.relative_signatures[0] }}
{% for signature in helper.relative_signatures[1:] %}
                 {{ signature }}
{% endfor %}
   :module: {{ helper.module }}
{% if helper.is_async %}
   :async:
{% endif %}

   {{ helper.docstring|indent(3) }}

   {% for definition in helper.definitions %}
   ``{{ definition.path }}:{{ definition.lineno }}``
   {% endfor %}

{% endfor %}
{% endif %}
