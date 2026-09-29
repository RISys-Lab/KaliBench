"use strict";

// The embedded data also powers the static, readable first example.
(() => {
  const dataNode = document.querySelector("#benchmark-examples-data");
  if (!dataNode) return;
  const { phases, examples } = JSON.parse(dataNode.textContent);
  const explorer = document.querySelector("#examples");
  const phaseSelect = explorer.querySelector("#example-phase");
  const dimensionSelect = explorer.querySelector("#example-dimension");
  const previous = explorer.querySelector("#example-previous");
  const next = explorer.querySelector("#example-next");
  const status = explorer.querySelector("#example-status");
  let selectedIndex = 0;
  let selectedComponent = "tool";

  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const code = (value, component) => {
    const node = element("code", String(value));
    if (component) node.dataset.annotation = component;
    return node;
  };

  function describeComponent(example) {
    const description = explorer.querySelector("#anatomy-description");
    const optional = Object.entries(example.optional_args);
    const values = optional.filter(([, value]) => value !== null);
    const text = {
      tool: ["Tool selection.", " The executable is ", example.tool_name, "."],
      keys: [
        "Optional keys.",
        optional.length
          ? " Highlighted flags select options. Names separated by | in the table are aliases for the same key."
          : " This example has no optional flags.",
      ],
      values: [
        "Optional values.",
        values.length
          ? " Highlighted values belong to their option keys. A null annotation means that a flag takes no value."
          : optional.length
            ? " These are standalone flags: their annotated values are null, so no value is written after them in the command."
            : " This example has no optional values.",
      ],
      arguments: [
        "Positional arguments.",
        example.positional_args.length
          ? " Highlighted arguments are listed below the command in their annotated order."
          : " This example has no positional arguments.",
      ],
    }[selectedComponent];
    description.replaceChildren(
      element("strong", text[0]),
      document.createTextNode(text[1]),
    );
    if (text[2])
      description.append(code(text[2]), document.createTextNode(text[3]));
  }

  function highlightComponent(announce = false) {
    explorer.querySelectorAll("[data-component]").forEach((button) => {
      const active = button.dataset.component === selectedComponent;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    explorer
      .querySelectorAll("[data-token], [data-annotation]")
      .forEach((node) => {
        node.classList.toggle(
          "is-emphasized",
          (node.dataset.token || node.dataset.annotation) === selectedComponent,
        );
      });
    describeComponent(examples[selectedIndex]);
    if (announce)
      status.textContent = explorer.querySelector(
        "#anatomy-description",
      ).textContent;
  }

  function renderExample(announce = true) {
    const example = examples[selectedIndex];
    const phase = phases.find((item) => item.id === example.phase);
    phaseSelect.value = example.phase;
    dimensionSelect.replaceChildren(
      ...examples
        .filter((item) => item.phase === example.phase)
        .map((item) => {
          const option = element(
            "option",
            item.dimension_short_label || item.dimension_label,
          );
          option.value = item.dimension;
          return option;
        }),
    );
    dimensionSelect.value = example.dimension;
    explorer.querySelector("#example-query").textContent = example.query;
    explorer
      .querySelector("#example-tool")
      .replaceChildren(code(example.tool_name, "tool"));
    const command = explorer.querySelector("#example-command");
    command.replaceChildren(
      ...example.tokens.map((token) => {
        if (!token.kind) return document.createTextNode(token.text);
        const node = element(
          "span",
          token.text,
          `command-token token-${{ tool: "tool", keys: "key", values: "value", arguments: "argument" }[token.kind]}`,
        );
        node.dataset.token = token.kind;
        return node;
      }),
    );
    command.parentElement.setAttribute(
      "aria-label",
      `Ground-truth command for ${example.tool_name}`,
    );

    const optional = Object.entries(example.optional_args);
    const tbody = explorer.querySelector("#example-option-rows");
    tbody.replaceChildren(
      ...optional.map(([key, value]) => {
        const row = document.createElement("tr");
        const keyCell = document.createElement("th");
        keyCell.scope = "row";
        keyCell.append(code(key.split("|").join(" | "), "keys"));
        const valueCell = document.createElement("td");
        valueCell.append(code(value === null ? "null" : value, "values"));
        if (value === null)
          valueCell.append(
            element("span", "Flag without a value", "null-explanation"),
          );
        row.append(keyCell, valueCell);
        return row;
      }),
    );
    explorer.querySelector("#example-options-table").hidden =
      optional.length === 0;
    explorer.querySelector("#example-no-options").hidden =
      optional.length !== 0;
    explorer.querySelector("#example-alias-note").hidden = !optional.some(
      ([key]) => key.includes("|"),
    );
    const positionals = explorer.querySelector("#example-positionals");
    if (example.positional_args.length) {
      const list = element("ol", undefined, "positional-values");
      example.positional_args.forEach((value) => {
        const item = document.createElement("li");
        item.append(code(value, "arguments"));
        list.append(item);
      });
      positionals.replaceChildren(list);
    } else
      positionals.replaceChildren(element("span", "None", "empty-arguments"));

    const source = explorer.querySelector("#example-source");
    source.textContent = `${example.split === "test" ? "Test" : "Training"} split · ${example.custom_id}`;
    source.href = `https://github.com/RISys-Lab/KaliBench/blob/main/${example.source_path}`;
    explorer.querySelector("#example-counter").textContent =
      `${selectedIndex + 1} / ${examples.length}`;
    previous.disabled = selectedIndex === 0;
    next.disabled = selectedIndex === examples.length - 1;
    highlightComponent();
    if (announce)
      status.textContent = `Example ${selectedIndex + 1} of ${examples.length}. ${phase.label}. ${example.dimension_label}. Tool: ${example.tool_name}.`;
  }

  phaseSelect.addEventListener("change", () => {
    selectedIndex = examples.findIndex(
      (item) => item.phase === phaseSelect.value,
    );
    renderExample();
  });
  dimensionSelect.addEventListener("change", () => {
    selectedIndex = examples.findIndex(
      (item) => item.dimension === dimensionSelect.value,
    );
    renderExample();
  });
  previous.addEventListener("click", () => {
    if (selectedIndex > 0) {
      selectedIndex -= 1;
      renderExample();
    }
  });
  next.addEventListener("click", () => {
    if (selectedIndex < examples.length - 1) {
      selectedIndex += 1;
      renderExample();
    }
  });
  explorer.querySelectorAll("[data-component]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedComponent = button.dataset.component;
      highlightComponent(true);
    });
  });
  renderExample(false);
  explorer.querySelectorAll("[data-example-enhanced]").forEach((node) => {
    node.hidden = false;
  });
})();
