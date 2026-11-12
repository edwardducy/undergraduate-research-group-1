#import "template/ceu-format.typ": ceu-format, appendices

#show: ceu-format.with(
  title: [$title$],
  authors: ($for(by-author)$"$it.name.literal$", $endfor$),
)
