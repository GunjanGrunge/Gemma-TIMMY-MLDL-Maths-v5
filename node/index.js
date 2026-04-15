"use strict";

const { spawnSync } = require("node:child_process");

function runPython(question, options = {}) {
  const python = options.python || process.env.MARTHA_V6_PYTHON || "python";
  const args = ["-m", "martha_v6", "--json", "--question", question];
  const result = spawnSync(python, args, {
    cwd: options.cwd || process.cwd(),
    encoding: "utf8",
    env: { ...process.env, ...(options.env || {}) },
    shell: process.platform === "win32",
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error((result.stderr || result.stdout || "martha_v6 failed").trim());
  }

  return parseStructuredOutput(result.stdout);
}

function parseStructuredOutput(text) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    throw new Error("martha_v6 returned empty output");
  }

  try {
    return JSON.parse(normalized);
  } catch (_) {
    const vm = require("node:vm");
    return vm.runInNewContext(`(${normalized})`);
  }
}

function answer(question, options = {}) {
  return runPython(question, options).output;
}

function answerStructured(question, options = {}) {
  return runPython(question, options);
}

module.exports = {
  answer,
  answerStructured,
};
