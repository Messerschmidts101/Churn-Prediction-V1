import React from 'react'

function Button({children, theme = 'primary', className, onClick}) {
    return (
        <button className={'btn btn-' +theme + " " + className} onClick={onClick}>{children}</button>
    )
}

export default Button