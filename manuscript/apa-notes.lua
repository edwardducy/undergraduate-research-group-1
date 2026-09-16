function Para(el)
  if not (quarto and quarto.doc and quarto.doc.is_format("pdf")) and not (FORMAT and FORMAT:match("latex")) then
    return el
  end

  if el.content and #el.content > 0 then
    local first = el.content[1]
    if first.t == "Emph" and #first.content > 0 and first.content[1].text == "Note." then
      table.remove(el.content, 1)
      if el.content[1] and el.content[1].t == "Space" then
        table.remove(el.content, 1)
      end
      local prefix = pandoc.RawInline("latex", "\\vspace{-8pt}{\\footnotesize\\singlespacing\\noindent\\textit{Note.} ")
      local suffix = pandoc.RawInline("latex", "\\par}\\vspace{6pt}")
      table.insert(el.content, 1, prefix)
      table.insert(el.content, suffix)
      return el
    end
  end
  return el
end
