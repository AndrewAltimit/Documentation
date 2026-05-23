#!/usr/bin/env ruby
# frozen_string_literal: true
#
# Math-rendering linter for the docs site.
#
# Catches the one class of regression that neither the Jekyll build nor
# html-proofer notices: equations that *render* incorrectly under MathJax.
# Specifically, inside display-math (`$$ ... $$`) it flags:
#   1. Unbalanced `$$` delimiters (odd count in a file).
#   2. Unicode math characters that should be LaTeX commands
#      (e.g. `ℏ`, `ψ`, `α`, `⟨`, `∂`, `²`, `ₙ` instead of \hbar, \psi, \alpha,
#      \langle, \partial, ^2, _n).
#   3. ASCII-art matrices drawn with pipes (e.g. `|0 1|`) instead of \pmatrix.
#
# Inline `$...$` is intentionally NOT scanned: a single `$` is ambiguous with
# currency (e.g. "$5/month" on the AWS pages), so scanning it would be noisy.
# The high-impact rendering problems (matrices, multi-symbol equations) all
# live in display math anyway.
#
# Stdlib-only, so it runs in a bare `ruby:3.2-slim` container with no bundle.
# Exits non-zero if any problem is found.

ROOT = File.expand_path(File.join(__dir__, "..", "github-pages"))

# Inside display math the only legitimate characters are ASCII — LaTeX writes
# every symbol as a command (\hbar, \psi, \langle, \partial, ^2, _n, \times).
# So any non-ASCII codepoint in a `$$ ... $$` span is a unicode-hack equation
# that should be converted. A couple of ASCII-safe typographic exceptions are
# allowed in case they appear inside \text{...} (none currently do).
def first_non_ascii(span)
  span.each_char.find { |c| c.ord > 0x7F }
end

# Pipe-delimited rows that look like an ASCII matrix, e.g. `|0 1|` or `|1 -1|`.
ASCII_MATRIX = /\|\s*-?\d[\d\s.\-]*\|/

errors = []

Dir.glob(File.join(ROOT, "**", "*.md")).reject { |p| p.include?("/vendor/") }.sort.each do |path|
  rel = path.sub(ROOT + "/", "")
  text = File.read(path)

  # Drop fenced code blocks so code never trips the math checks.
  body = text.gsub(/```.*?```/m, "")

  dd = body.scan(/\$\$/).length
  if dd.odd?
    errors << "#{rel}: unbalanced `$$` display-math delimiters (found #{dd}, expected an even count)"
    next # span scan below would be meaningless with an odd count
  end

  # Each `$$ ... $$` display span (single- or multi-line).
  body.scan(/\$\$(.+?)\$\$/m) do |(span)|
    one_line = span.gsub(/\s+/, " ").strip
    snippet = one_line.length > 80 ? "#{one_line[0, 77]}..." : one_line
    # Text-mode content is prose and may legitimately contain accents/units
    # (e.g. \text{Schrödinger}, \text{Å}); strip it before the symbol check.
    math_only = span.gsub(/\\(?:text|textrm|textbf|textit|textsf|mathrm|mbox|operatorname)\s*\{[^{}]*\}/, "")
    if (c = first_non_ascii(math_only))
      errors << "#{rel}: unicode `#{c}` (U+#{format('%04X', c.ord)}) inside display math (use a LaTeX command): $$ #{snippet} $$"
    end
    if span.match(ASCII_MATRIX)
      errors << "#{rel}: ASCII-art matrix inside display math (use \\begin{pmatrix}): $$ #{snippet} $$"
    end
  end
end

if errors.empty?
  puts "math-lint: OK — no display-math rendering problems found"
  exit 0
else
  warn "math-lint: #{errors.length} problem(s) found\n\n"
  warn errors.join("\n")
  exit 1
end
