import React from 'react'

function Button({children, theme = 'primary', onClick}) {
    return (
        <button className={'btn btn-' +theme} onClick={onClick}>{children}</button>
    )
}

export default Button