#!/usr/bin/env node
"use strict";

const { answerStructured } = require("./index");

function parseArgs(argv) {
  const questions = [];
  let json = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--json") {
      json = true;
      continue;
    }
    if (arg === "--question") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("--question requires a value");
      }
      questions.push(value);
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!questions.length) {
    throw new Error("Pass at least one --question");
  }
  return { questions, json };
}

function main() {
  const { questions, json } = parseArgs(process.argv.slice(2));
  questions.forEach((question, index) => {
    const result = answerStructured(question);
    if (json) {
      process.stdout.write(`${JSON.stringify(result)}\n`);
      return;
    }
    process.stdout.write(`${"=".repeat(80)}\n`);
    process.stdout.write(`Question ${index + 1}: ${question}\n`);
    process.stdout.write(`${"-".repeat(80)}\n`);
    process.stdout.write(`${result.output}\n\n`);
  });
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exit(1);
}
