import React from 'react'

function Label({forLabel, children}) {
  return (
    <label htmlFor={forLabel}>{children}</label>
  )
}

export default Label