import React from 'react'

function Label({forLabel, children}) {
  return (
    <label for={forLabel}>{children}</label>
  )
}

export default Label