import React from 'react'

function Form({id, className, children, action, method}) {
    return (
        <form id={id} className={className} action={action} method={method}>{children}</form>
    )
}

export default Form